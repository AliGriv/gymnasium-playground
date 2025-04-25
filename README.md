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
    gym-playground <project-set-name> <sub-project-name>
    ```

## Toy Text

### Taxi

- Documentation is [here](https://gymnasium.farama.org/environments/toy_text/taxi/)

- See all the options:
```sh
gym-playground toy-text taxi --help
```

- Example: Train for 15000 episodes, and test on one
