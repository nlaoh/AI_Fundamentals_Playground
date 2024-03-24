import math

class PriorityQueue:
    def __init__(self):
        self.queue = list()

    def isEmpty(self):
        return len(self.queue) == 0

    def append(self, data):
        self.queue.append(data)

    def exist(self, data):
        return data in self.queue

    def poll(self):
        min = 0

        for i in range(len(self.queue)):
            if self.queue[i].f < self.queue[min].f:
                min = i

        item = self.queue[min]
        del self.queue[min]

        return item

    def remove(self, data):
        self.queue.remove(data)

class Edge:
    def __init__(self, target, weight):
        self.target = target
        self.weight = weight

class Node:
    def __init__(self, name, x, y) -> None: #None is not required
        self.name = name
        self.x = x
        self.y = y
        # parameters for A* search
        self.g = 0 #cost so far
        self.h = 0 #heuristic estimate
        self.f = 0 #total estimated cost
        
        self.neighbours = list()

        self.parent = None    

    def __hash__(self):
        return hash(self.f)
    
def heuristic(node1, node2):
    result = math.sqrt((node1.x - node2.x) * (node1.x - node2.x) + (node1.y - node2.y) * (node1.y - node2.y))
    return result

def search(source, destination):
    explored = set()
    priority_queue = PriorityQueue()

    priority_queue.append(source)


    while not priority_queue.isEmpty():
        current = priority_queue.poll()

        explored.add(current)

        # Found our destination, return the path
        if current.x == destination.x and current.y == destination.y:
            return destination

        for edge in current.neighbours:
            child = edge.target
            cost = edge.weight
            g = current.g + cost
            f = g + heuristic(current, destination)            

            # Keep looking if f value is larger, we are looking for smaller f values
            if child in explored and f >= child.f:
                continue
            # Otherwise if have not visited or the f(x) is smaller
            elif not priority_queue.exist(child) or f < child.f:
                # Found a good node
                
                child.parent = current
                child.g = g
                child.f = f

                # instead of updating the child on priority
                # but rather remove and reinsert
                if priority_queue.exist(child):
                    priority_queue.remove(child)

                priority_queue.append(child)

n1 = Node('A', 0, 0)
n2 = Node('B', 10, 20)
n3 = Node('C', 20, 40)
n4 = Node('D', 30, 10)
n5 = Node('E', 40, 30)
n6 = Node('F', 50, 10)
n7 = Node('G', 50, 40)

n1.neighbours.append(Edge(n2, 10))
n1.neighbours.append(Edge(n4, 50))

n2.neighbours.append(Edge(n3, 10))
n2.neighbours.append(Edge(n4, 20))

n3.neighbours.append(Edge(n5, 10))
n3.neighbours.append(Edge(n7, 30))

n4.neighbours.append(Edge(n6, 80))

n5.neighbours.append(Edge(n6, 50))
n5.neighbours.append(Edge(n7, 10))

n7.neighbours.append(Edge(n6, 10))

path = search(n1, n6)

def display(path):
    print('--- print path ---')
    print('[', end='')

    while True:
        print(path.name + ' ', end='')
        path = path.parent
        if(path == None):
            break
        
    print(']')

display(path)