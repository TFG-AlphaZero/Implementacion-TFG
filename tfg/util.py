from functools import reduce

from joblib import Parallel, delayed

from tfg.strategies import HumanStrategy
from tfg.games import WHITE, BLACK


def play(game, white, black, games=1, max_workers=None,
         render=False, print_results=False):
    """Play n games of the provided game where players are using strategies
    white and black, respectively.

    Args:
        game (tfg.games.GameEnv): Game to be played.
        white (tfg.strategies.Strategy): Strategy for WHITE player.
        black (tfg.strategies.Strategy): Strategy for BLACK player.
        games (int): Number of games that will be played.
            If max_workers is None they will be played iteratively.
            Otherwise, games / max_workers will be played iteratively by each
            worker. Defaults to 1.
        max_workers (int): If set, maximum number of processes that will be
            launched to play simultaneously. Not recommended if one of the
            players is tfg.strategies.HumanStrategy. Defaults to None.
        render (bool): Whether to render the game after every turn or not.
            Defaults to False.
        print_results (bool): Whether to print the results at the end of each
            game. Defaults to False.

    Returns:
        (int, int, int): Cumulative results of all games in the format
            (WHITE wins, draws, BLACK wins).

    """

    def play_(g):
        def print_winner():
            if not print_results:
                return
            game.render(mode='human')
            winner = game.winner()
            if winner == 0:
                print("DRAW")
            else:
                print(f"PLAYER {'1' if winner == 1 else '2'} WON")

        def get_winner_index():
            winner = game.winner()
            return {1: 0, 0: 1, -1: 2}[winner]

        def move(obs):
            this = players[game.to_play]
            other = players[-game.to_play]
            action = this.move(obs)
            next_obs, _, done, _ = game.step(action)
            this.update(action)
            other.update(action)
            return next_obs, done

        results = [0, 0, 0]
        players = {WHITE: white, BLACK: black}
        for _ in range(g):
            observation = game.reset()
            white.update(None)
            black.update(None)
            if render and not isinstance(white, HumanStrategy):
                game.render()

            while True:
                observation, done = move(observation)
                if done:
                    results[get_winner_index()] += 1
                    print_winner()
                    break
                elif (render and
                      not isinstance(players[game.to_play], HumanStrategy)):
                    game.render()

        return tuple(results)

    if max_workers is None:
        return play_(games)

    n_games = get_games_per_worker(games, max_workers)

    results = Parallel(max_workers)(delayed(play_)(g) for g in n_games)
    return reduce(lambda acc, x: map(sum, zip(acc, x)), results)


def get_games_per_worker(games, max_workers):
    """Calculates the number of games each processor should play.

    Args:
        games (int): Total number of games to be played.
        max_workers (int): Number of processors that will play those games.

    Returns:
         list[int]: Number of games each processor should play. The list
            contains one element per processor.

    """
    d_games = games // max_workers
    r_games = games % max_workers
    n_games = [d_games] * max_workers
    if r_games != 0:
        for i in range(r_games):
            n_games[i] += 1
    return n_games


def enable_gpu():
    """Utility method that sets memory growth for all GPUs.

    Otherwise they would not work.
    """
    import tensorflow as tf

    # GPU didn't work otherwise
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
