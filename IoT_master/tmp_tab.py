# Example list with reserved desks - ONLY FOR TEST PURPOSES!!!
# If you would like to change this remember that the
# number of list's elements must be the same
# as number of desks in config.yml file!!!
from tools.person_detect import *

detected = detection()
with torch.no_grad():
    while True:
        occupancy = detected.detect()
        print(occupancy)
#desk_reservations = [0, 1, 0, 1, 1]
