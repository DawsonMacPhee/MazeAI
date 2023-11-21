import numpy
import os

# Based on A* algorithm found in https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2

class PathFinder():
    def __init__(self, maze):
        self.setMaze(maze)
        
    def findEnds(self):
        foundStart = False
        foundGoal = False
        for i in range(0,len(self.maze)):
            for j in range(0,len(self.maze[0])):
                if not foundStart and self.maze[i][j] == 3:
                    self.start_pos = Node(position=(i,j))
                    self.start_pos.g = self.start_pos.h = self.start_pos.f = 0
                    foundStart = True
                    if foundGoal:
                        return
                elif not foundGoal and self.maze[i][j] == 4:
                    self.goal_pos = Node(position=(i,j))
                    self.goal_pos.g = self.goal_pos.h = self.goal_pos.f = 0
                    foundGoal = True
                    if foundStart:
                        return
    
    def setMaze(self, maze):
        self.maze = maze
        self.findEnds()
        self.map_width = len(self.maze[0])
        self.map_height = len(self.maze)
                    
    def findPath(self):
        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(self.start_pos)

        # Loop until you find the end
        while len(open_list) > 0:
            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == self.goal_pos:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1] # Return reversed path

            # Generate children
            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] > (len(self.maze) - 1) or node_position[0] < 0 or node_position[1] > (len(self.maze[len(self.maze)-1]) -1) or node_position[1] < 0:
                    continue

                # Make sure walkable terrain
                if self.maze[node_position[0]][node_position[1]] == 0:
                    continue

                # Create new node
                new_node = Node(node_position,parent=current_node)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:

                # Child is on the closed list
                if child in closed_list:
                    continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - self.goal_pos.position[0]) ** 2) + ((child.position[1] - self.goal_pos.position[1]) ** 2)
                child.f = child.g + child.h

                in_open = False
                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        in_open = True
                        continue
                
                if in_open:
                    continue
                
                # Add the child to the open list
                open_list.append(child)
      
class Node():
    def __init__(self, position=None, *, parent=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position