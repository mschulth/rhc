from exp.Exploration import Exploration, MaxEpisodesReachedException


class RandomExploration(Exploration):

    def run(self):
        self._init()

        for i in range(self.n_episodes):
            self.logger.info("Exploration episode {}".format(i+1))
            a, s, x, y = self.env.execute_rand(self.horizon, reset=True)
            try:
                self._end_episode(x, y, s, a)
            except MaxEpisodesReachedException:
                pass
