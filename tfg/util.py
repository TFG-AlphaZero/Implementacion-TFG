from functools import reduce

from joblib import Parallel, delayed

from tfg.strategies import HumanStrategy


def play(game, s1, s2, games=1, max_workers=None, render=False, print_results=False):
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

        results = [0, 0, 0]
        for _ in range(g):
            observation = game.reset()
            if render and not isinstance(s1, HumanStrategy):
                game.render()

            while True:
                action = s1.move(observation)
                observation, _, done, _ = game.step(action)
                if done:
                    results[get_winner_index()] += 1
                    print_winner()
                    break
                elif render and not isinstance(s2, HumanStrategy):
                    game.render()
                action = s2.move(observation)
                observation, _, done, _ = game.step(action)
                if done:
                    results[get_winner_index()] += 1
                    print_winner()
                    break
                elif render and not isinstance(s1, HumanStrategy):
                    game.render()

        return tuple(results)

    if max_workers is None:
        return play_(games)

    d_games = games // max_workers
    r_games = games % max_workers
    n_games = [d_games] * max_workers
    if r_games != 0:
        for i in range(r_games):
            n_games[i] += 1

    results = Parallel(max_workers)(delayed(play_)(g) for g in n_games)
    return reduce(lambda acc, x: map(sum, zip(acc, x)), results)
