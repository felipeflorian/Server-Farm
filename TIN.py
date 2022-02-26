#######################################################################
#             Computational and differential Geometry                 #
#                         First Homework                              #
#                 Andres Felipe Florian Quitian                       #
#######################################################################

import numpy as np 
import math as m
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy import sparse
from scipy.spatial import distance


class TIN:

    """ Class TIN that represents a triangulated irregulater network

            Attributes: - Points (coordinates). 
                        - Elevations
                        - Delaunay triangulation with respect to the points.
    """

    def __init__(self, points, elevations):

        """ Class attributes initiliazation """

        self.pt = points #Coordinates
        self.el = elevations #Elevations
        self.tri = Delaunay(points) #Elevations
    
    def plot_initiliazation(self):

        """ Method that initilizes the plot in 3D """

        self.fig = plt.figure()
        self.ax = self.fig.gca(projection ='3d')

    def plot_tri_2D(self, title = 'Triangulation plot in 2D'):

        """ Plots the triangulation in 2D, also initializes self.fig and self.ax """

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_title(title)
        self.ax.triplot(self.pt[:, 0], self.pt[:, 1], self.tri.simplices, alpha=0.3)
        self.ax.plot(self.pt[:, 0], self.pt[:, 1], 'o', c = 'blue', alpha = 0.3)

    def plot_tri_3D(self, vertex_color=False, title = 'Triangulation plot in 3D'):

        """ Plots the triangulation in 3D the parameter vertex_color 
            determines how the vertices will be shown if its true they would be plot with 3 colors
            depending of the elevation """

        points = np.array([self.pt[:,0], self.pt[:,1], self.el]).T #Coordinates and its elevation 
        edges = self.obtain_edges() 

        #Variables Initiliazation
        x = np.array([])
        y = np.array([])
        z = np.array([])

        #Edges plot
        for (i,j) in edges:
            x = np.append(x, [points[i, 0], points[j, 0], np.nan])      
            y = np.append(y, [points[i, 1], points[j, 1], np.nan])      
            z = np.append(z, [points[i, 2], points[j, 2], np.nan])
        
        self.ax.set_title(title)
        self.ax.plot3D(x, y, z, color='black', lw='0.4', alpha = 0.5)
        
        if vertex_color:
            #Plot vertes with different colors
            col = np.array(['blue','red','green'])
            self.ax.scatter(points[:,0], points[:,1], points[:,2], c=col[self.get_classification()])
    
    def elevation_plot(self, title = 'Elevation plot'):

        """ Plots the triangulation in 3D as a 'heat map' """
        
        T = mtri.Triangulation(self.pt[:,0], self.pt[:,1], triangles = self.tri.simplices)
        self.ax.set_title(title)
        col_bar = self.ax.plot_trisurf(T, self.el, cmap = 'coolwarm')
        bar = self.fig.colorbar(col_bar)

    def get_classification(self):

        """ Determines the classification that will be used for the plotting in the method plot_tri_3D()
            if condition holds """

        h = (max(self.el)- min(self.el))/3 
        values = [min(self.el)+h*i for i in range(4)] #'Interval' partition
        pos = [0]*len(self.el)

        #Elevation classification
        for i in range(3):
            pos = np.where(np.logical_and(values[i] <= self.el, self.el <= values[i+1]), i, pos) 
        return pos

    def obtain_edges(self):

        """ Method that returns the edges of the vertices """

        edges = set()
        for (i0, i1, i2) in self.tri.simplices:
            edges.add(tuple_surt(i0,i1))
            edges.add(tuple_surt(i0,i2))
            edges.add(tuple_surt(i1,i2))
        return edges

    def interpolation_(self, pt):

        """ Method that returns the elevation of the given point pt
            
            Input: -pt: An array
            Output: Elevation of the point pt"""


        v, e, index = self.in_triangle(pt) #Find the triangle where pt lies
        result = LinearNDInterpolator(v, values=e) #LinearInterpolation
        ele = result(pt)
        return ele[0]

    def in_triangle(self, pt):

        """ Evaluates if a point pt is in a triangle

            Input: - pt: An array 
            Output: - Vertices of the triangle, its elevations and the index of the simplex"""
        
        index_ = self.tri.find_simplex(pt) 
        simplice = self.tri.simplices[index_]
        v = np.array([self.pt[simplice[0]],self.pt[simplice[1]],self.pt[simplice[2]]])
        e = np.array([self.el[simplice[0]],self.el[simplice[1]],self.el[simplice[2]]])
        return v, e, index_   

    def triangle_area(self, v):

        """ Gives the area of a triangle with vertices v

            Input: -v: Array with the triangle vertices
            Ouput: Triangle areas"""

        A = np.transpose(np.array([v[:,0], v[:,1], [1,1,1]]))
        return (1/2)*np.linalg.det(A)

    def angles(self, v):

        """ Angles between three vertices

            Input: -v: Array with the vertex coordinates
            Output: Array with the angles between the vertices in orden"""

        A = v[0]
        B = v[1]
        C = v[2]

        AB = B-A
        AC = C-A
        BA = A-B
        BC = C-B
        CA = A-C
        CB = B-C

        n_AB = np.linalg.norm(AB)
        n_AC = np.linalg.norm(AC)
        n_BC = np.linalg.norm(BC)
    
        angle_1 = np.degrees(np.arccos(np.dot(AB,AC)/(n_AB*n_AC)))
        angle_2 = np.degrees(np.arccos(np.dot(BA,BC)/(n_AB*n_BC)))
        angle_3 = np.degrees(np.arccos(np.dot(CA,CB)/(n_AC*n_BC)))

        return np.array([angle_1, angle_2, angle_3])

    def get_distances(self):
        
        """ Returns a matrix with the euclidean distance 
            between each pair of vertices"""

        A = np.zeros((len(self.pt), len(self.pt)))
        points = self.pt.tolist()

        #Calculates the distance and stored it in A
        for a in self.tri.simplices:
            pos_1, pos_2, pos_3 = a[0], a[1], a[2]
            v_1 = points[pos_1]
            v_2 = points[pos_2]
            v_3 = points[pos_3]
            d_1 = distance.euclidean(v_1, v_2)
            d_2 = distance.euclidean(v_1, v_3)
            d_3 = distance.euclidean(v_2, v_3)
            A[pos_1, pos_2] = d_1
            A[pos_2, pos_1] = d_1
            A[pos_1, pos_3] = d_2
            A[pos_3, pos_1] = d_2
            A[pos_2, pos_3] = d_3
            A[pos_3, pos_2] = d_3
        return A

    def elevation(self, pt, plot = False):

        """ Elevation of a point

            Input: -pt: Point as an array
            Output: Elevation value, and plot if needed"""

        v, e, index = self.in_triangle(pt)
        ele = self.interpolation_(pt)
        if plot:
            title = 'Plot of the triangulation and the given point'
            self.plot_tri_3D(title = title)
            self.ax.scatter(pt[0], pt[1], ele, marker = '*', c = 'green', linewidths = 1.5, label = 'Given point')
            self.fig.legend(loc ='lower right')
        return ele

    def largest_drainage(self, pt, plot = False):

        """ Find the largest drainage basin for a given point

            Input: -pt: Array with the coordinates
                   -plot: Boolean. Determines if the plot is needed
            Output: Returns the points with the corresponding drainage basin and its area"""

        v, e, index_ = self.in_triangle(pt) #Triangle where the point lies
        area = self.triangle_area(v) #Area of the triangle

        #Neighbors of the triangle where pt lies
        neighbors_in = self.tri.neighbors[index_] 
        neighbors = self.tri.simplices[neighbors_in]

        areas = np.zeros(len(neighbors))

        #Total area between the initial triangle and each one of its neighbors
        for i, simp in enumerate(neighbors):
            t = np.array([self.pt[simp[0]],self.pt[simp[1]],self.pt[simp[2]]])
            area_neighbor = self.triangle_area(t)
            total_area = area + area_neighbor
            areas[i] = total_area

        #Find the maximum area found it and the cuadrilateral between the triangles
        maximum= np.argmax(areas)
        opti_ = neighbors[maximum]
        opti_neighbor = self.pt[opti_]
        drainage = np.unique(np.concatenate((v, opti_neighbor), axis=0), axis=0)
        
        if plot:
            #Find elevation for the given point and the vertices of the cuadrilateral for plotting
            ele = self.interpolation_(pt)
            vertices = [self.interpolation_([i,j]) for (i,j) in drainage]
            self.plot_tri_3D(title = 'Largest drainage basin for the given point')
            self.ax.scatter(pt[0], pt[1], ele, marker = '*', c = 'green', label = 'Given point')
            self.ax.scatter(drainage[:,0], drainage[:,1], vertices, marker = '*', c = 'red', linewidths = 0.1, label = 'Cuadrilateral Vertices')
            self.fig.legend(loc = 'lower right')
        return drainage, areas[maximum], vertices

    def sampling_accuracy(self, plot = False):

        """ Accuracy of the sampling performed on the terrain, it means
            find the max and min angles of the triangulation

            Input: -plot: Boolean. Determines if the plot is needed
            Output: Maximum and minimal angles of the triangulation"""

        #Find the angles for each triangle and stored its maximum and minimum
        max_angles = []
        min_angles = []
        
        for a in self.tri.simplices:
            v = np.array([self.pt[a[0]], self.pt[a[1]], self.pt[a[2]]])
            tri_angle = self.angles(v)
            max_angles.append(max(tri_angle))
            min_angles.append(min(tri_angle))
        
        maximum = max(max_angles)
        minimum = min(min_angles)

        if plot:
            #Get the vertex triangles for plotting
            index_max = max_angles.index(maximum)
            index_min = min_angles.index(minimum)
            t1 = self.tri.simplices[index_max]
            t2 = self.tri.simplices[index_min]
            max_triangle = np.array([self.pt[t1[0]], self.pt[t1[1]], self.pt[t1[2]]])
            min_triangle = np.array([self.pt[t2[0]], self.pt[t2[1]], self.pt[t2[2]]])
            a_max = self.angles(max_triangle).tolist()
            a_min = self.angles(min_triangle).tolist()
           
            #Takes the specific vertes where the max and min was found
            ver_max = max_triangle[a_max.index(maximum)]
            ver_min = min_triangle[a_min.index(minimum)] 
            self.plot_tri_2D(title = 'Sampling accuracy')
            self.ax.plot(ver_max[0], ver_max[1], marker = '*', c = 'red', label = 'Maximum angle')
            self.ax.plot(ver_min[0], ver_min[1], marker = '*', c = 'yellow', label = 'Minimum angle')
            self.fig.legend(loc = 'upper right')
        return maximum, minimum 

    def euclidean_tree(self, plot=False):

        """ Found the minimun expansion tree with respect to the euclidean distance
            
            Input: -plot: Boolean. For plotting or not
            Output: Euclidean minimum spanning tree"""

        A = self.get_distances() #Obtain the distances
        tree = minimum_spanning_tree(sparse.csr_matrix(A)) #Expansion tree

        if plot:
            tree = tree.toarray()
            vertices = []
            pos = 0

            #Found the adjacent vertices
            for i in tree:
                indices = [j for j in range(len(i)) if i[j] != 0]
                vertices = vertices + [(pos,k) for k in indices if (pos,k) not in vertices or (k,pos) not in vertices] 
                pos += 1
            
            #Plotting
            self.plot_tri_2D(title = 'Euclidean minimum spanning tree')
            for i in vertices:
                v1 = self.pt[i[0]]
                v2 = self.pt[i[1]]
                x = [v1[0], v2[0]]
                y = [v1[1], v2[1]]
                
                self.ax.plot(x, y, c = 'red')
                self.ax.plot(v1[0], v1[1], marker = 'o', c = 'red')
                self.ax.plot(v2[0], v2[1], marker = 'o', c = 'red')
        return tree
        

#Function that its used in the method obtain_edges
def tuple_surt(a,b):
    """ Function that return a tuple in orden """
    return (a,b) if a < b else (b,a)

#Example 

#Read the file where the data is
f = open("pts30c.dat", "r")
str_numbers = []
for x in f:
  str_numbers.append(x.split())

pt = []
ele = []
for i in str_numbers:
    pt.append((float(i[0]),float(i[1])))
    ele.append(float(i[2]))

coords = np.array(pt)
elevs = np.array(ele)

test = TIN(coords, elevs)

#Point for points 2 and 3 
p = [12, -5]

###### Point 1

test.plot_initiliazation()
test.elevation_plot()
#test.plot_tri_3D(True)

###### Point 2

test.plot_initiliazation()
elevation_ = test.elevation(p, True)
print('The elevation for the point {0} is {1}'.format(p, elevation_))

###### Point 3

test.plot_initiliazation()
drainage, area, elvs = test.largest_drainage(p, True)
print('The largest drainage basin for the point {0} is: {1}'.format(p, drainage))
print('With elevations {0}: '.format(elvs))
print('and its area is {0}'.format(area))

###### Point 4

max, min = test.sampling_accuracy(True)
print('The sampling accuracy has  {0} and {1} as maximum and minimum angles respectively'.format(max, min))

###### Point 5

test.euclidean_tree(True)


plt.show()