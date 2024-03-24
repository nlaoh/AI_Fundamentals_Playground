class Node:
    def __init__(self, name):
        self.name = name
        self.visited = False
        self.neighbours = list()

def backtrace(parent, start, end):
    path = [end]
    
    while path[-1].name != start.name:
        path.append(parent[path[-1]])
        
    path.reverse()
    
    return path
# 
def traverse(root, end):
    parent = dict()
    
    # Treat like a stack
    stack = list()

    stack.append(root)

    root.visited = True

    # 
    while len(stack) > 0: 
        currentNode = stack.pop()  #main difference from BFS, for DFS pop the last element
        
        if currentNode.name == end.name:
            return backtrace(parent, root, end)
        
        currentNode.visited = True

        for adjacent_neighbour in currentNode.neighbours:
            if not adjacent_neighbour.visited:
                parent[adjacent_neighbour] = currentNode
                stack.append(adjacent_neighbour)

def display(path):
    print('Path: ', end='')
    for node in path:
        print(node.name + ' ', end='')

a = Node('A')
b = Node('B')
c = Node('C')
d = Node('D')
e = Node('E')
        
a.neighbours.append(b)
a.neighbours.append(c)
c.neighbours.append(d)
d.neighbours.append(b)
d.neighbours.append(e)

display(traverse(a, d))