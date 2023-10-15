import numpy as np
import random
size = 7

def negamax(table, depth, alpha, beta, color):


    if depth == 0 or (checkWin(table) != 0):
        g = Graph(size * size + 2)
        g.graph = makeGraph(table)
        heuristic = g.dijkstra(0, size * size + 1)
        return color * heuristic

    childNodes = generateMoves(table, color)

    value = -np.inf
    for child in childNodes:
        value = max(value, -negamax(child, depth-1, -alpha, -beta, -color))
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    return value


def generateMoves(table, color):
    listOfMoves = []
    for i in range(size):
        for j in range(size):
            if table[i][j] == '_':
                table[i][j] = 'R' if color == 1 else 'B'
                listOfMoves.append(table)
                table[i][j] = '_'

    return listOfMoves

def adjacentFind(x, y):
    adj_list = [[x - 1, y + 1],
                [x, y + 1],
                [x + 1, y],
                [x + 1, y - 1],
                [x, y - 1],
                [x - 1, y]]

    adj_list = [nei for nei in adj_list if 0 <= nei[0] <= size - 1 and 0 <= nei[1] <= size - 1]
    return adj_list


def convert(num):
    x = int((num - 1) / size)
    y = int((num - 1) % size)
    list = [x, y]
    return list


def makeGraph(arr):
    graph = [[1000 for x in range(size * size + 2)] for y in range(size * size + 2)]

    for j in range(1, size * size + 1):
        coordinatej = convert(j)
        if (arr[coordinatej[0]][coordinatej[1]] == 'B'):
            continue
        if (coordinatej[0] == 0):
            if (arr[coordinatej[0]][coordinatej[1]] == 'R'):
                graph[0][j] = 0
            else:
                graph[0][j] = 1

        if (coordinatej[0] == size - 1): graph[j][size * size + 1] = 0
    for i in range(1, size * size + 1):
        coordinatei = convert(i)
        if arr[coordinatei[0]][coordinatei[1]] == 'B':
            continue
        for j in range(1, size * size):
            coordinatej = convert(j)
            for v in adjacentFind(coordinatei[0], coordinatei[1]):
                if coordinatej[0] == v[0] and coordinatej[1] == v[1]:
                    if arr[coordinatej[0]][coordinatej[1]] == 'R':
                        graph[i][j] = 0
                    elif arr[coordinatej[0]][coordinatej[1]] == '_':
                        # temp = 0
                        for w in adjacentFind(coordinatej[0], coordinatej[1]):
                            if arr[w[0]][w[1]] == 'B':
                                graph[i][j] = 5
                            else:
                                graph[i][j] = 10
    return graph


def checkWin(table):
    g = Graph(size * size + 2)
    g.graph = makeGraph(table)
    if g.dijkstra(0, size * size + 1) == 1000: return 1
    if g.dijkstra(0, size * size + 1) == 0: return -1
    return 0


class Graph:

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    def minDistance(self, dist, sptSet):

        min_ = np.inf

        for u in range(self.V):
            if dist[u] < min_ and sptSet[u] == False:
                min_ = dist[u]
                min_index = u

        return min_index

    def dijkstra(self, src, sink):

        dist = [np.inf] * self.V
        dist[src] = 0
        sptSet = [False] * self.V

        for cout in range(self.V):

            x = self.minDistance(dist, sptSet)

            sptSet[x] = True

            for y in range(self.V):
                if self.graph[x][y] > -1 and sptSet[y] == False and \
                        dist[y] > dist[x] + self.graph[x][y]:
                    dist[y] = dist[x] + self.graph[x][y]

        return 1000 - dist[sink]


def printTable(table):
    clear()
    print("----------------------------------------------------------")
    spCounter = 2
    print(end='     ')
    for i in range(size):
        print(i, end=" ")
    print()
    for i in range(size):
        print(spCounter * " ", size-1 - i, end=" ")
        for j in range(size):
            print(table[i][j], end=" ")
        print(size-1 - i)
        spCounter += 1
    print(end='            ')
    for i in range(size):
        print(i, end=" ")
    print()

    print("----------------------------------------------------------")


def clear():
    for i in range(30):
        print()


def main():

    table = []

    for i in range(size):
        table.append(["_" for j in range(size)])

    personWin = False
    agentWin = False

    str1 = input("would you like to go first? :(Y,N)\n")
    turn = 0 if str1.upper() == 'Y' else 1
    printTable(table)
    while not (personWin or agentWin):
        if turn % 2 == 0:
            x, y = list(map(int, input(
                "person (BLUE) connects vertically.\nagent (RED) connects horizontal.\nPerson turn : \nEnter "
                "the coordinate x and y:").split()))
            table[size-1 - x][y] = "B"
            printTable(table)
            if checkWin(table) == -1:
               personWin = True

        else:
            print("person (BLUE) connects vertically.\nagent (RED) connects horizontal.\nagent ...")
            maxValue = -np.inf
            tableValue = [[-np.inf for x in range(size)] for y in range(size)]
            for i in range(size):
                for j in range(size):
                    if table[i][j] == '_':
                        table[i][j] = 'R'
                        tableValue[i][j] = negamax(table, 1, -np.inf, np.inf, 1)
                        table[i][j] = '_'
                        maxValue = max(maxValue, tableValue[i][j])

            listOfCoordinates = []
            for i in range(size):
                for j in range(size):
                    if maxValue == tableValue[i][j]:
                        listOfCoordinates.append((i,j))
            print(tableValue)
            x,y = random.choice(listOfCoordinates)
            print(maxValue)

            table[x][y] = 'R'
            printTable(table)

            if checkWin(table) == 1:
                agentWin = True
        turn += 1

    if personWin:
        print('You Won')
    elif agentWin:
        print('You Lost')


if __name__ == '__main__':
    main()
