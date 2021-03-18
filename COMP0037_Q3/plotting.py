# This module contains some plotting routines for the state value function and the policy
# One small trick is that, by default, matplot rotates the grid to use image-style format
# The code here reverses this, but it's a bit of a pain with indices and just required
# a lot of trial and error.

import numpy as np
import matplotlib.pyplot as plt

from robot_enums import Heading
from robot_enums import action_string

# Helper to print all dimensions of an array
def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])

# Helper to flip the array to an image which is rendered in the proper orientation
def to_image(array):
    img = np.flip(np.transpose(array), 0)
    return img

class ValueFunctionPlotter(object):

    def __init__(self):
        self.opened_figure = False

    def plot(self, value_function):

        # Extract the value functions for different robot orientations
        north_value_function = value_function[:,:, Heading.NORTH]
        east_value_function = value_function[:,:, Heading.EAST]
        south_value_function = value_function[:,:, Heading.SOUTH]
        west_value_function = value_function[:,:, Heading.WEST]

        # Create the figure if it isn't present already
        if self.opened_figure is False:
            self.plot_first_figure(north_value_function, east_value_function, south_value_function, west_value_function)
            return

        # Update the images
        self.im_north = self.axs[0,0].imshow((north_value_function), origin='lower')
        self.im_east = self.axs[0,1].imshow((east_value_function), origin='lower')
        self.im_south = self.axs[1,0].imshow((south_value_function), origin='lower')
        self.im_west = self.axs[1,1].imshow((west_value_function), origin='lower')

        # Update the labels
        for i in range(north_value_function.shape[0]):
            for j in range(north_value_function.shape[1]):
                self.text[0][i][j].set_text(north_value_function[i, j])
                self.text[1][i][j].set_text(east_value_function[i, j])
                self.text[2][i][j].set_text(south_value_function[i, j])
                self.text[3][i][j].set_text(west_value_function[i, j])

        # Update the plot and inject a small pause to let the GUI process all the events
        plt.draw()
        plt.pause(0.001)


    # First time the plot has been c
    def plot_first_figure(self, north_value_function, east_value_function, south_value_function, west_value_function):

        # Create and label the plots
        self.fig, self.axs = plt.subplots(2,2)
        self.axs[0,0].set_title("North Heading")
        self.axs[0,1].set_title("East Heading")
        self.axs[1,0].set_title("South Heading")
        self.axs[1,1].set_title("West Heading")

        # Render the value functions
        self.im_north = self.axs[0,0].imshow((north_value_function), origin='lower')
        self.im_east = self.axs[0,1].imshow((east_value_function), origin='lower')
        self.im_south = self.axs[1,0].imshow((south_value_function), origin='lower')
        self.im_west = self.axs[1,1].imshow((west_value_function), origin='lower')

        self.fig.suptitle('State Value Functions')
        self.fig.tight_layout()
        plt.ion()

        # Loop over data dimensions and create text annotations.
        self.text=[[[0 for _ in range(north_value_function.shape[1])] for _ in range(north_value_function.shape[0])] for _ in range(4)]
        print(dim(self.text))
        print(north_value_function.shape[1])
        for i in range(north_value_function.shape[0]):
            for j in range(north_value_function.shape[1]):
                self.text[0][i][j] = self.axs[0,0].text(j, i, north_value_function[i, j],
                                     ha="center", va="center", color="w")
                self.text[1][i][j] = self.axs[0,1].text(j, i, east_value_function[i, j],
                                     ha="center", va="center", color="w")
                self.text[2][i][j] = self.axs[1,0].text(j, i, south_value_function[i, j],
                                     ha="center", va="center", color="w")
                self.text[3][i][j] = self.axs[1,1].text(j, i, west_value_function[i, j],
                                     ha="center", va="center", color="w")
                 
        plt.show()
        self.opened_figure = True



class PolicyPlotter(object):

    def __init__(self):
        self.opened_figure = False

    def plot(self, policy):

        # Extract the value functions for different robot orientations
        north_policy = policy[:,:, Heading.NORTH]
        east_policy = policy[:,:, Heading.EAST]
        south_policy = policy[:,:, Heading.SOUTH]
        west_policy = policy[:,:, Heading.WEST]

        # Generate the output figure if needed
        if self.opened_figure is False:
            self.plot_first_figure(north_policy, east_policy, south_policy, west_policy)
            return

        # Update the images
        self.im_north = self.axs[0,0].imshow(north_policy, origin='lower')
        self.im_east = self.axs[0,1].imshow(east_policy, origin='lower')
        self.im_south = self.axs[1,0].imshow(south_policy, origin='lower')
        self.im_west = self.axs[1,1].imshow(west_policy, origin='lower')

        # Update the labels
        for i in range(north_policy.shape[0]):
            for j in range(north_policy.shape[1]):
                self.text[0][i][j].set_text(action_string(north_policy[i, j]))
                self.text[1][i][j].set_text(action_string(east_policy[i, j]))
                self.text[2][i][j].set_text(action_string(south_policy[i, j]))
                self.text[3][i][j].set_text(action_string(west_policy[i, j]))

        # Plot the results
        plt.draw()
        plt.pause(0.001)


    # First update we have to create a bunch of stuff
    def plot_first_figure(self, north_policy, east_policy, south_policy, west_policy):
        
        self.fig, self.axs = plt.subplots(2,2)

        self.axs[0,0].set_title("North Heading")
        self.axs[0,1].set_title("East Heading")
        self.axs[1,0].set_title("South Heading")
        self.axs[1,1].set_title("West Heading")

        # Render the value functions
        self.im_north = self.axs[0,0].imshow(north_policy, origin='lower')
        self.im_east = self.axs[0,1].imshow(east_policy, origin='lower')
        self.im_south = self.axs[1,0].imshow(south_policy, origin='lower')
        self.im_west = self.axs[1,1].imshow(west_policy, origin='lower')

        self.fig.suptitle('Policy')
        self.fig.tight_layout()
        plt.ion()

        # Loop over data dimensions and create text annotations.
        self.text=[[[0 for _ in range(north_policy.shape[1])] for _ in range(north_policy.shape[0])] for _ in range(4)]
        print(dim(self.text))
        print(north_policy.shape[1])
        for i in range(north_policy.shape[0]):
            for j in range(north_policy.shape[1]):
                self.text[0][i][j] = self.axs[0,0].text(j, i, action_string(north_policy[i, j]),
                                     ha="center", va="center", color="w")
                self.text[1][i][j] = self.axs[0,1].text(j, i, action_string(east_policy[i, j]),
                                     ha="center", va="center", color="w")
                self.text[2][i][j] = self.axs[1,0].text(j, i, action_string(south_policy[i, j]),
                                     ha="center", va="center", color="w")
                self.text[3][i][j] = self.axs[1,1].text(j, i, action_string(west_policy[i, j]),
                                     ha="center", va="center", color="w")
                 
        plt.show()
        self.opened_figure = True
