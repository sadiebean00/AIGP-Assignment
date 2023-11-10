class Node(object):
    """docstring for Node"""

    def __init__(self, parent=None, pos=None, goal=None, biggerN=1):
        super(Node, self).__init__()
        self.goal = goal
        self.parent = parent
        self.position = pos
        self.n = biggerN
        self.f = 0
        self.child = []
        if self.parent:
            self.n = parent.n + 1
        self.f = self.n + self.distance(self, self.goal)

    def __eq__(self, other):
        return self.position == other.position

    @staticmethod
    def distance(self, goal):
        return ((self.position[0]-goal[0])**2)+((self.position[1]-goal[1])**2)


def aStar(grid, start, end):
    openedList = []
    closedList = []
    visited = []
    newPos = None
    path = []

    if start == end:
        print("start = end!")
        return [start.position]
    openedList.append(start)
    visited.append(start.position)
    currentNode = openedList[0]
    openedList.remove(currentNode)
    while True:
        if currentNode == end:
            break
        for x, y in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            newPos = [currentNode.position[0] + x, currentNode.position[1] + y]
            if newPos[0] > (len(grid) - 1) or newPos[0] < 0 or newPos[1] > (len(grid[len(grid)-1]) - 1) or newPos[1] < 0:
                continue
            if grid[newPos[0]][newPos[1]] != 0:
                continue

            if newPos not in visited:
                if abs(x) == abs(y):
                    openedList.append(Node(currentNode, newPos, currentNode.goal, 1.4))
                else:
                    openedList.append(Node(currentNode, newPos, currentNode.goal))
                visited.append(newPos)

        openedList.sort(key=lambda x: x.f)
        if len(openedList) >= 1:
            currentNode = openedList.pop(0)
            closedList.append(currentNode)
        else:
            return -1

    try:
        while True:
            path.append(currentNode.position)
            currentNode = currentNode.parent
    except Exception:
        return path[::-1]


def main():
    grid = [
        [3, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    end = Node(None, [5, 5], [5, 5])
    start = Node(None, [0, 0], end.position)
    print(aStar(grid, start, end))


if __name__ == '__main__':
    main()

