from tfg.strategies import HumanStrategy


def play(game, s1, s2, games=1, render=False, print_results=False):
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
    for _ in range(games):
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
