import taichi as ti
import math as math

ti.init(debug=False,arch=ti.cpu)
real = ti.f32 #data type f32 -> float in C

max_num_particles = 1000
lambda_epsilon = 0.0 #user specified relaxation parameter(important) -> adjustable
dt = 1e-2#simulation time step(important) -> adjustable
dt_inv = 1 / dt
dx = 0.02
dim = 2
pbd_num_iters = 30#Iteration number(important) -> adjustable

scalar = lambda: ti.var(dt=real) #2D dense tensor
vec = lambda: ti.Vector(dim, dt=real) #2*1 vector(each element in a tensor)
mat = lambda: ti.Matrix(dim, dim, dt=real) #2*2 matrix(each element in a tensor)

num_particles = ti.var(ti.i32, shape=())
paused = ti.var(ti.i32, shape=())
damping = ti.var(ti.f32, shape=())

particle_mass = 1 #mi
particle_mass_inv = 1 / particle_mass # 1 / mi
particle_mass_invv = 1 / (particle_mass_inv + particle_mass_inv + lambda_epsilon)
maximum_constraints = 50

bottom_y = 0.05
bottom_x = 0.95
epsolon = 0.0001 #digit accurary(important) -> adjustable

x, v, old_x = vec(), vec(), vec()
actuation_type = scalar()
total_constraint = ti.var(ti.i32, shape=())
# rest_length[i, j] = 0 means i and j are not connected
rest_length = scalar()
position_delta_tmp = vec()
position_delta_sum = vec()
constraint_neighbors = ti.var(ti.i32)
constraint_num_neighbors = ti.var(ti.i32)

gravity = [0, -9] #direction
H_force = [0, 0] #another gr

@ti.layout  #Environment layout(placed in ti.layout) initialization of the dimensiond of each tensor variables(global)
def place():
    ti.root.dense(ti.ij, (max_num_particles, max_num_particles)).place(rest_length)
    ti.root.dense(ti.i, max_num_particles).place(x, v, old_x, actuation_type, position_delta_tmp, position_delta_sum) #initialzation to zero
    nb_node = ti.root.dense(ti.i, max_num_particles)
    nb_node.place(constraint_num_neighbors)
    nb_node.dense(ti.j, maximum_constraints).place(constraint_neighbors)

@ti.kernel
def old_posi(n: ti.i32):
    for i in range(n):
        old_x[i] = x[i]

@ti.kernel
def find_constraint(n: ti.i32):
    for i in range(n):
        nb_i = 0
        for j in range(n):
            if rest_length[i, j] != 0: #spring-constraint
                x_ij = x[i] - x[j]
                dist_diff = abs(x_ij.norm() - rest_length[i, j])
                if dist_diff >= epsolon:
                    constraint_neighbors[i, nb_i] = j
                    nb_i += 1
        constraint_num_neighbors[i] = nb_i

@ti.kernel
def substep(n: ti.i32, t: ti.i32): # Compute force and new velocity
    for i in range(n):
        v[i] *= ti.exp(-dt * damping[None]) # damping
        total_force = ti.Vector(gravity) * particle_mass #gravity -> accelaration
        if actuation_type[i] == 1:
            #total_force = ti.Vector([9.8 * t, 0], real) * particle_mass
            total_force = ti.Vector(H_force) * particle_mass
        v[i] += dt * total_force / particle_mass

@ti.kernel
def collision_check(n: ti.i32):# Collide with ground
    for i in range(n):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0
        if x[i].y > 1 - bottom_y:
            x[i].y = 1 - bottom_y
            v[i].y = 0
        if x[i].x > bottom_x:
            x[i].x = bottom_x
            v[i].x = 0
        if x[i].x < 1 - bottom_x:
            x[i].x = bottom_x
            v[i].x = 0

@ti.kernel
def Position_update(n: ti.i32):# Compute new position
    for i in range(n):
        x[i] += v[i] * dt

@ti.kernel
def stretch_constraint(n: ti.i32):
    for i in range(n):
        pos_i = x[i]
        posi_tmp = ti.Vector([0.0, 0.0])
        for j in range(constraint_num_neighbors[i]):
            p_j = constraint_neighbors[i, j]
            pos_j = x[p_j]
            x_ij = pos_i - pos_j
            dist_diff = x_ij.norm() - rest_length[i, p_j]
            grad = x_ij.normalized()
            position_delta = -particle_mass * particle_mass_invv * dist_diff * grad / constraint_num_neighbors[i]
            posi_tmp += position_delta
        position_delta_tmp[i] = posi_tmp

@ti.kernel
def apply_position_deltas(n: ti.i32):
    for i in range(n):
        x[i] += position_delta_tmp[i]

@ti.kernel
def updata_velosity(n: ti.i32): #updata velosity after combining constraints
    for i in range(n):
        v[i] = (x[i] - old_x[i]) * dt_inv

@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32): # Taichi doesn't support using Matrices as kernel arguments yet
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1

@ti.kernel
def new_costraint(p1_index: ti.i32, p2_index: ti.i32, dist: ti.f32):
    # Connect with existing particles
    rest_length[p1_index, p2_index] = dist
    rest_length[p2_index, p1_index] = dist

def check_single_particle():
    cons = rest_length.to_numpy()
    invalid_particle = []
    for i in range(num_particles[None]):
        sum = 0
        for j in range(num_particles[None]):
            sum += cons[i ,j]
        if sum == 0:
            invalid_particle.append(i)
    return invalid_particle

def forward(n, time_step):
    #the first three steps -> only consider external force
    old_posi(n)
    substep(n, time_step)
    Position_update(n)
    #print(x.to_numpy()[0:n] - old_x.to_numpy()[0:n])
    collision_check(n)
    #print(x.to_numpy()[0:n])
    #add constraints
    for i in range(pbd_num_iters):
        constraint_neighbors.fill(-1)
        find_constraint(n)
        #print("This is ", i , "th iteration.")
        stretch_constraint(n)
        apply_position_deltas(n)
        collision_check(n)
    updata_velosity(n)

gui = ti.GUI('Mass Spring System', res=(640, 640), background_color=0xdddddd)

class Scene:
    def __init__(self, resolution):
        self.n_particles = 0
        self.resolution = resolution
        self.actuation = []
        self.x = []

    def add_line(self, x_start, x_end, y):
        w = x_end - x_start
        w_count = int(w / dx) * self.resolution  # N particles in each grid(N = 1,2,3,4,5,..)
        if w_count == 0: #tangent line
            self.x.append([
                (x_start + x_end) / 2,
                y
            ])
            self.n_particles += 1
            self.actuation.append(0)
        else:
            real_dx = w / w_count  # step
            flag = False
            for i in range(w_count):
                if flag == False and 0.3 < x_start < 0.5 and 0.4 < y < 0.6:
                    flag = True
                    self.actuation.append(1)
                else:
                    self.actuation.append(0)
                self.x.append([  # each particle's position
                    x_start + (i + 0.5) * real_dx,  # 0.5 * real_dx is very small
                    y
                ])
                self.n_particles += 1

    def finalize(self):
        global max_num_particles
        max_num_particles = self.n_particles #the true number of particles
        #50 for manually added particles by mouse-click
        print('maximum number of particles:', max_num_particles)

def x_position_circle(center_x, center_y, radius, y):
    x_start = center_x - math.sqrt(abs(radius ** 2 - (y - center_y) ** 2))
    x_end = center_y + math.sqrt(abs(radius ** 2 - (y - center_y) ** 2))
    return x_start, x_end

def robot_circle(scene, center_x, center_y, radius): #the confiurtion of the soft robot?
    h = 2 * radius
    h_count = int(h / dx) * scene.resolution
    real_dy = h / h_count
    for j in range(h_count + 1):
        y = center_y - radius + j * real_dy
        x_start, x_end = x_position_circle(center_x, center_y, radius, y)
        scene.add_line(x_start, x_end, y)

damping[None] = 30

def main():
    #Read all mesh points from txt.file
    points = []
    with open('./Contour_points/vertex.txt', 'r') as f:
        data = f.readlines()
    for line in data:
        odom = line.split()
        points.append([float(odom[0]), float(odom[1])])
    constraints = []
    with open('./Contour_points/constraint.txt', 'r') as f:
        data = f.readlines()
    for line in data:
        odom = line.split()
        constraints.append([float(odom[0]), float(odom[1]), float(odom[2])])
    num_particles[None] = 0
    for i in points:
        new_particle(i[0], i[1])
    for i in constraints:
        new_costraint(int(i[0]), int(i[1]), i[2])

    single_particle_list = check_single_particle()
    n = num_particles[None]
    time_step = 0
    index = 0
    while True:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == gui.SPACE:
                paused[None] = not paused[None]

        index += 1
        if index % 2 == 0:
            if time_step <= 14:
                time_step += 1

        if not paused[None]:
            for step in range(2):
                forward(n, time_step)

        X = x.to_numpy()
        for i in range(n):
            # if scene.actuation[i] == 1:
            #     gui.circles(X[i:i+1], color=0x33bb76, radius=5)
            # else:
            if i not in single_particle_list:
                gui.circles(X[i:i+1], color=0xffaa77, radius=5)
        gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
        for i in range(num_particles[None]):
            for j in range(i + 1, num_particles[None]):
                if rest_length[i, j] != 0:
                    gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
        gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
        gui.show()

if __name__ == '__main__':
    main()
