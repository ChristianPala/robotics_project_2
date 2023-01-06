import numpy as np
import roboticstoolbox as rtb
import depth_support as ds

# Generate a stick-robot with function LRMate200iD4S_gen(). The robot is defined using Denavit-Hartenberg convention
lrmate = ds.LRMate200iD4S_gen()
# See the addendum on Denavit-Hartenberg convention to know more.

# Dynamic visualization
lrmate.teach()
# NB: the teach panel can freeze the console! Try ctrl+c after closing the window, or restart the kernel

# We can compute the complete forward kinematics, or all the “partial” kinematics up to the k-th joint
lrmate.fkine([0, 0, 0, 0, -np.pi/2, 0])
lrmate.fkine_all([0, 0, 0, 0, -np.pi/2, 0])

# Due to how the gearing of joints 2 and 3 is realized, for each degree of positive rotation of joint 2 also joint 3
# gets rotated of the same degrees in its positive direction, although this does not appear in joint coordinates.

# Function joints_fanuc2corke allows to transform joint angles logged from the Fanuc LRMate200iD/4s (in degrees) to
# the correct angles to be provided to its roboticstoolbox model (in radians) for making it reproduce the same movement
# in 3D space.

# Let's try this with few sets of joint positions
joint_angles = np.zeros([10, 6])
joint_angles[:, 4] = -90
joint_angles[:, 1] = np.linspace(0, 45, 10)
# Correction due to this robot's transmission (j2-j3 interaction) and conversion to radians
joint_angles_mod = ds.joints_fanuc2corke(joint_angles)
lrmate.plot(joint_angles_mod)