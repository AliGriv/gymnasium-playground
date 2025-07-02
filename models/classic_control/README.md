# Notes About Models

## Mountain Car with Continous Actions

- Model File Name: `mountain_car_cont.pkl`
- Parameters:
    ```json
    {
    "cont_actions": true,
    "num_actions_bins": 25,
    "num_position_bins": 20,
    "num_velocity_bins": 20
    }
    ```
- This model was not trained in one training sessions. Several trials were required to get to this stage.
 Training a model for Mountain Car, with continuous actions, has not been implemented efficiently.

- It took me at least 4 times, each time at least 5000 episodes.