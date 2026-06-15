uint8 BOTTLE=1
uint8 PERSON=2
uint8 MOUSE=3
uint8 BOWL=4
uint8 BANANA=5

# fields
uint8 identifier
float32 confidence

float32 x_min
float32 y_min
float32 x_max
float32 y_max

float32 error_x
float32 error_y

# tracking data
int32 track_id
# velocity pixels/s
float32 v_x
float32 v_y

float32 duration
