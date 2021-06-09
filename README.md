# Deep Learning Applied to Turn-based Board Games
This project is part of the Final Degree Project for Computer Science at Universidad Complutense de Madrid (UCM).

[AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go) is a general-purpose reinforcement learning algorithm developed by [DeepMind](https://deepmind.com/) in 2017. It can learn from _tabula rasa_, given no domain knowledge except the game rules, and achieves superhuman performance in combinational games such as Go and Chess. It uses self-play, so that it starts playing randomly against itself and gradually learns further comprehension of the game.

In this project, we develop our own version of AlphaZero, capable of being executed on a personal computer. 

Due to the lack of powerful computational resources, we have focused in less complex games such as Tic-Tac-Toe and Connect 4. However, our implementation is highly versatile and it can be used for any game very easily. In order to verify that our implementation is learning properly, we have tested it against other implemented algorithms (Minimax and Monte Carlo Tree Search).

## Getting Started 🚀

The repository is divided into multiple folders:
 - [tfg](https://github.com/TFG-AlphaZero/Implementacion-TFG/tree/master/tfg): the core of the project is in this folder. The implementation is divided into several modules: game representation, strategies, neural network and the actual AlphaZero. 
 - [game](https://github.com/TFG-AlphaZero/Implementacion-TFG/tree/master/game): games implementation are found in this folder.
 - [models](https://github.com/TFG-AlphaZero/Implementacion-TFG/tree/master/models): in this folder we store some neural network models and checkpoints for saving and testing purposes.
 - [experiments](https://github.com/TFG-AlphaZero/Implementacion-TFG/tree/master/experiments): in this folder we keep the Jupyter Notebooks that were used for testing the implementation.

### Prerequisites 📋

See [Requirements](https://github.com/TFG-AlphaZero/Implementacion-TFG/blob/master/requirements.txt).

## Running the code ⚙️

You can see an example of execution in the Jupyter Notebooks included in the [experiments](https://github.com/TFG-AlphaZero/Implementacion-TFG/tree/master/experiments) folder.

## Authors ✒️

  - **Pablo Sanz Sanz** - Project Member - [Soy-yo](https://github.com/Soy-yo)
  - **Juan Carlos Villanueva Quirós** - Project Member - [jcturing](https://github.com/jcturing)
  - **Antonio A. Sánchez Ruiz-Granados** - Project Manager - [antsanchucm](https://github.com/antsanchucm)

## License 📄

This project is licensed under the [MIT License](https://github.com/TFG-AlphaZero/Implementacion-TFG/blob/master/LICENSE).
