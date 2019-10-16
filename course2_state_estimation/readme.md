# State Estimation

All python scripts are run/executed by
`python3 python_file.py`

## Lesson1.py
Very basic least squares estimating resistance given current and voltage data.

## Lesson2.py
Batch least squares implementation using the same data in lesson1.py

## Lesson3.py
One iteration EKF given a distance to a landmark and the height of the landmark. This was part of the video lecture, cant remember if it was an actual assignment.

## Lesson6_example.py
Same data as lesson3.py however now using a UKF instead of a EKF.

## Lesson6.py
During lesson5 and lesson6 videos the instructor shows the difference between a UKF and EKF using range and angle measurements. This script is my attempt at implementing the UKF solution on recreated data. The actual data was never provided, so this script randomly generates a number of samples in the same range that is presented in the video.

## week2_assignment/week2_assignment.py
Offline solution to the week2 programming assignment. The code was copied from the provided jupyter notebook and pasted here. Made testing and development offline easier.
