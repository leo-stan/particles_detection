# particle_dataset

This dataset contains data from a Velodyne HDL-32E Lidar and  Multisense S21 stereo camera in five different outdoor scenes with airborne particles of dust and fog.

Lidar points are labeled between particles and non-particles.

### Download link

Link: https://cloudstor.aarnet.edu.au/plus/s/lxvbgCiFSgtcdkj

Password: Particle_dataset1

### Folder structure

training/lidar
training/label

validation/lidar
validation/label

testing/lidar
testing/label

### Loading files in python

- lidar raw scans: numpy.fromfile("lidar_file.bin")
- labels: open("label_file.txt", 'r')

### Labels

- 0: hard surface
- 1: smoke
- 2: dust
