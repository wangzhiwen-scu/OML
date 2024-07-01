from tqdm import tqdm
import time

# Sample range values, adjust as needed.
outer_loop_range = 5
inner_loop_range = 10

# Outer loop with tqdm.
for i in tqdm(range(outer_loop_range), desc="Outer loop", position=0):
    
    # Some code...
    time.sleep(0.1)
    
    # Inner loop with tqdm.
    for j in tqdm(range(inner_loop_range), desc="Inner loop", position=1, leave=False):
        
        # More code...
        time.sleep(0.1)
