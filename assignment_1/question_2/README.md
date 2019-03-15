# Computational Graph Usage - Gravity Simulator

This problem helps to understand the concept of computational graphs and their uses. It also gives an idea about how tensorflow take advantage of parallel computation.

## Problem Statement

Given masses, intial positions and initial velocites of 100 particles, you are required to find the final positions and velocities of particles when the minimum distance between any pair of particles falls below a given threshold. The only working forces are Newtonian Gravitational Forces, in a two-dimensional rectangular coordinate system.

### Requriements

Tested on Ubuntu 18.04 with:   

```  
  jupyter notebook: 4.4.0  
  numpy: 1.16.2  
  python: 3.6.7  
  tensorflow: 1.13.1  
  tornado: 6.0  
```

## Running and understanding the code
To run the code open jupyter notebook and open the file. Now from menu bar select cell->run all option.
The input files are provided in q2_input folder. These files are loaded incode by these lines:  

```
# load data from external files  
posi = np.load('q2_input/positions.npy')  
mass = np.load('q2_input/masses.npy')  
velo = np.load('q2_input/velocities.npy')  
```
The computational graph is then prepared using the above as initial variable.

```
# create graph  
graph = tf.get_default_graph()  
with graph.as_default():  
    thresh = tf.constant(.1, dtype=tf.float64, name='Treshold')  
    G = tf.constant(667000.0, dtype=tf.float64, name='Gravitation_constant')  
    dt = tf.constant(0.0001, dtype=tf.float64, name='Delta_t')  
    m = tf.Variable(mass, dtype=tf.float64, name='Masses')  
    v = tf.Variable(velo, dtype=tf.float64, name='Velocities')  
    p = tf.Variable(posi, dtype=tf.float64, name='Positions')  
    size_m = tf.size(m)  
    rel = tf.cast(tf.Variable(tf.fill([size_m, size_m, 2],0.0)), dtype=tf.float64, name='Rel_pos')  
```

After this, the tensor variables are manipulated according to Newton's laws of mostion.
To write the graph summary in logs to have a visual representation in Tensorboard below code is used:  
At very top to include timestamp in log directory name

```
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
```

And just after the computational graph is construceted

```
summary = tf.summary.scalar('v', v)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
```

There is a session loop and the final output is stored in the end.
