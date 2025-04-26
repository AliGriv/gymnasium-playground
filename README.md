# gymnasium-playground
A personal playground for learning basic reinforcement learning with the Gymnasium library.

## Setup

1. Create Virtual Environment

```sh
python3 -m venv venv
```
2. Activate the virtual environment and install the requirements

```sh
pip3 install -e .
```

3. Run the project similar to this:


    a. To see all the project sets:

    ```sh
    gym-playground --help
    ```

    b. To see each project's available sub-projects:
    ```sh
    gym-playground <project-set-name> --help
    ```

    c. To see all available options for a certain project:
    ```sh
    gym-playground <project-set-name> <sub-project-name> --help
    ```

## Toy Text

### Taxi

- Documentation is [here](https://gymnasium.farama.org/environments/toy_text/taxi/)

- See all the options:
```sh
gym-playground toy-text taxi --help
```
```
Usage: gym-playground toy-text taxi [OPTIONS]

  Run the Taxi experiment

Options:
  --train                 Run training mode
  --test                  Run test mode
  --model-save-path TEXT  Where to save the pickel model (used in training)
  --model-load-path TEXT  Path to pickel load model from, in test-only mode it
                          is required
  --render                Render the environment
  --learning-rate FLOAT   Learning rate
  --epsilon FLOAT         Starting epsilon (will be 0 in test mode)
  --epsilon-decay FLOAT   Epsilon decay rate (default: epsilon / (episodes /
                          2))
  --episodes INTEGER      Number of episodes to run  [required]
  --plot                  Plot some statistics from training procedure
  --help                  Show this message and exit.
```

- Example: Train for 15000 episodes, and save the model:

```sh
gym-playground toy-text taxi --train --model-save-path model/taxi --episodes 15000
```

- Example: Test the model for 2 episodes and render:

```sh
gym-playground toy-text taxi --test --model-load-path model/taxi.pkl --episodes 2 --render 
```

