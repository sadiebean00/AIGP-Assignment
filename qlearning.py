import numpy as np
import random
import time


def maze_generator(N):
    # This generates a random maze, where the S represents the start point and the E represents the end point.
    # There is also a '#' where the wall's are represented alongside '.' which represents spaces where you can move.
    random_Row = N
    random_Column = N
    steps = N

    i = 0
    j = 0

    maze_grid = [[0 for x in range(random_Column)] for y in range(random_Row)]
    maze_grid[i][j] = 'S'

    while steps != 0:
        i_or_j = random.choice([True, False])
        inc_or_dec = random.randint(0, 5)
        if i_or_j and (inc_or_dec > 0) and i != random_Row - 1 and maze_grid[i + 1][j] != 'S' and maze_grid[i + 1][j] \
                != '.':
            i = i + 1
            maze_grid[i][j] = '.'
            if steps == 1:
                maze_grid[i][j] = 'E'
        elif i_or_j == False and (inc_or_dec > 0) and j != random_Column - 1 and maze_grid[i][j + 1] != 'S' and \
                maze_grid[i][j + 1] != '.':
            j = j + 1
            maze_grid[i][j] = '.'
            if steps == 1:
                maze_grid[i][j] = 'E'
        elif i_or_j and (inc_or_dec == 0) and i != 0 and maze_grid[i - 1][j] != 'S' and maze_grid[i][j - 1] != '.':
            i = i - 1
            maze_grid[i][j] = '.'
            if steps == 1:
                maze_grid[i][j] = 'E'
        elif i_or_j == False and (inc_or_dec == 0) and j != 0 and maze_grid[i][j - 1] != 'S' and maze_grid[i][
            j - 1] != '.':
            j = j - 1
            maze_grid[i][j] = '.'
            if steps == 1:
                maze_grid[i][j] = 'E'
        else:
            continue
        steps = steps - 1
    ii = 0
    jj = 0

    for ii in range(0, random_Row):
        for jj in range(0, random_Column):
            i_or_j = random.choice([True, False])
            if maze_grid[ii][jj] != 'S':
                if maze_grid[ii][jj] != '.':
                    if maze_grid[ii][jj] != 'E':
                        if i_or_j:
                            maze_grid[ii][jj] = '#'
                        else:
                            maze_grid[ii][jj] = '.'
    return maze_grid


def get_end_point(maze, N):
    # This gets the end point to be at a random point in the maze. The start point will always remain the same.
    s = 0
    for i in range(N):
        for j in range(N):
            if maze[i][j] == 'E':
                return s
            s += 1
    return -1


def get_next_point(maze, N):
    # This gets the next point where the next state of the maze is represented in a 2d matrix [][]
    index = np.zeros([N, N])
    s = 0
    for i in range(N):
        for j in range(N):
            index[i][j] = s
            s += 1
    next_point = np.zeros([N * N, 4])
    s = 0
    for i in range(N):
        for j in range(N):
            if (i - 1 < 0 or maze[i - 1][j] == '#'):
                next_point[s][0] = s
            else:
                next_point[s][0] = index[i - 1][j]

            if (j + 1 > N - 1 or maze[i][j + 1] == '#'):
                next_point[s][1] = s
            else:
                next_point[s][1] = index[i][j + 1]

            if (i + 1 > N - 1 or maze[i + 1][j] == '#'):
                next_point[s][2] = s
            else:
                next_point[s][2] = index[i + 1][j]

            if (j - 1 < 0 or maze[i][j - 1] == '#'):
                next_point[s][3] = s
            else:
                next_point[s][3] = index[i][j - 1]

            if (maze[i][j] == '#' or ((i - 1 < 0 or maze[i - 1][j] == '#') and (j + 1 > N - 1 or maze[i][j + 1] == '#')
                                      and (i + 1 > N - 1 or maze[i + 1][j] == '#') and
                                      (j - 1 < 0 or maze[i][j - 1] == '#'))):
                next_point[s][0] = -1
                next_point[s][1] = -1
                next_point[s][2] = -1
                next_point[s][3] = -1

            if (maze[i][j] == 'E'):
                next_point[s][0] = s
                next_point[s][1] = s
                next_point[s][2] = s
                next_point[s][3] = s

            s += 1
    return next_point


def reward_eval(policy, reward, next_point, V_old, discount_factor=1.0, theta=0.00001):
    # This looks at the policy for the grid and decides on the best action to take.
    V_new = np.zeros(16)

    for s in range(16):
        v = 0.0

        for a, action_prob in list(enumerate(policy[s])):
            next = next_point[s][a]

            if (next != -1):
                v += action_prob * (reward + discount_factor * V_old[int(next)])
        V_new[s] = v

    return np.array(V_new)


def best_action(a):
    # This function returns the best action that could be taken by the index. It also changes the frequency.
    if np.array_equal(a, [0, 0, 0, 0]) or np.array_equal(a, [1, 0, 0, 0]) or np.array_equal(a, [0, 1, 0, 0]) or \
            np.array_equal(a, [0, 0, 1, 0]) or np.array_equal(a, [0, 0, 0, 1]):
        return np.argmax(a)

    freq = np.zeros(4)
    i = 0

    while i < 4:
        j = 0
        while j < 4:
            if abs(a[i] - a[j] < 0.00001):
                freq[i] += 1
            j += 1
        i += 1
    max_index = np.argmax(a)

    if freq[max_index] > 2:
        return -1
    return np.argmax(a)


def is_deterministic(policy):
    # This checks whether or not the policy created above is deterministic and if so, what to do.
    rows = policy.shape[0]
    columns = policy.shape[1]

    for x in range(0, rows):
        for y in range(0, columns):
            if abs(policy[x, y] - 0.25) < 0.001:
                return False
    return True


def reward_improvement(reward, next_point, goal_index, reward_eval_fn=reward_eval, discount_factor=1.0):
    # This evaluates the rewards and keeps on iterating through the policy/reward until the optimal(final) policy
    # is found.
    def one_step_ahead(state, V):
        # This function helps to calculate the value for the action in a given state.
        A = np.zeros(4)
        i = 0
        for a in range(4):
            next = next_point[state][a]
            if (next != -1):
                A[i] += (reward + discount_factor * V[int(next)])
            i = i + 1
        return A

    policy = np.ones([16, 4]) / 4
    policy[goal_index] = np.zeros(4)
    initial_policy = policy.copy()

    V_old = np.zeros(16)
    V_new = np.zeros(16)

    k = 0
    while True:
        print("Iteration ", k, ":")
        policy[goal_index] = np.zeros(4)
        policy_old = policy.copy()

        V_new = reward_eval_fn(initial_policy, reward, next_point, V_old)
        V_old = V_new.copy()
        print("Current Values: ")
        print(V_new)
        policy_stable = True

        for s in range(16):
            action_values = one_step_ahead(s, V_new)

            best_a = best_action(action_values)

            if (best_a == 1):
                policy_stable = False
            if (best_a != -1):
                policy[s] = np.eye(4)[best_a]

        k += 1
        print("Current Policy Probability distribution: ")
        print(policy)

        if np.array_equal(policy, policy_old) and k > 1 and is_deterministic(policy):
            return (policy, V_new)


def value_iterations(nS, goal_index, discount_factor=1.0, theta=0.0001):
    # This is where the values iterate to our needs.
    def one_step_ahead(state, V):
        A = np.zeros(4)
        i = 0
        for a in range(4):
            next = next_state[s][a]
            if (next != -1):
                A[i] += (reward + discount_factor * V[int(next)])
            i = i + 1
        return A

    V_old = np.zeros(nS)
    V_new = np.zeros(nS)
    while True:
        delta = 0
        V_old = V_new.copy()

        for s in range(nS):
            if s == goal_index:
                continue

            A = one_step_ahead(s, V_old)
            best_action_value = np.max(A)

            delta - max(delta, np.abs(best_action_value - V_old[s]))

            V_new[s] = best_action_value
        print(V_new)

        if delta < theta:
            break

    policy = np.zeros([nS, 4])
    for s in range(nS):
        A = one_step_ahead(s, V_new)
        best_action = np.argmax(A)

        policy[s, best_action] = 1.0

    return policy, V_new


def get_path(p):
    # This gets the path from the start to the end and also lists the actions taken to get from the Start to the
    # End. It'll list the path in numbers, as well as through directions (bottom, right, up, and left)
    finished = False
    path = []
    actions = []
    next_square = 0
    current_square = 0
    while finished == False:
        finished = True
        for i in range(4):
            if p[next_square][i] == 1:
                finished = False
                if i == 0:
                    next_square -= N
                    actions.append("up")
                elif i == 1:
                    next_square += 1
                    actions.append("right")
                elif i == 2:
                    next_square += N
                    actions.append("bottom")
                else:
                    next_square -= 1
                    actions.append("left")
                path.append(next_square)
    return (path, actions)


def get_max_value_state(next_states, V):
    # This gets the maximum value for the state and allows the index to get to that maximum value.
    max = -10000000000000
    index = 0
    for ii in range(4):
        c_val = V[int(next_states[ii])]
        if c_val > max:
            max = c_val
            index = ii
    return ii


def get_path_value(V, next_point):
    # This gets the value of the path that the route is taking. This is pretty much similar to the normal get_path.
    finished = False
    path = []
    actions = []
    next_square = 0
    current_square = 0
    while finished == False:
        finished = True
        for i in range(4):
            new_next_square = get_max_value_state(next_point[next_square], V)
            if new_next_square != next_square:
                finished = False
                next_square = new_next_square
                path.append(next_square)
                if i == 0:
                    actions.append("up")
                elif i == 1:
                    actions.append("right")
                elif i == 2:
                    actions.append("bottom")
                else:
                    actions.append("left")
    return path, actions


if __name__ == '__main__':
    # This is where the N value is set, alongside the maze generator being called. The reward and discount factor are
    # also called here. This is also where the display for the maze's outcome are shown/called.
    N = 4
    maze = maze_generator(N)
    reward = -1
    discount_factor = 1.0
    goal_index = get_end_point(maze, N)
    next_state = list(get_next_point(maze, N))

    for ii in range(N):
        print(maze[ii])

    start_time = time.time()
    policy, v = reward_improvement(reward, next_state, goal_index)
    exec_time = (time.time() - start_time)
    print("\n\n-------------Final Results---------\n\n-")
    print("Policy Probability Distribution:")
    print(policy)
    print("")
    print("Value Function:")
    print(v)
    print("")
    path, actions = get_path(policy)
    print("Path: ")
    print(path)
    print("Actions: ")
    print(actions)
    print("--- Running Time : %s seconds ---" % exec_time)
