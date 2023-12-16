# """
# Many changes made specially the plots, but some of the code and ideas belong to:
# @misc{youtube_dzskvsszgjs,
#     author = "{Robert Cowher - DevOps, Python, AI}",
#     title = "{Playing Breakout with Deep Reinforcement Learning}",
#     year = {2023},
#     howpublished = "\url{https://www.youtube.com/watch?v=DzSKVsSzGjs}",
# }
# """

import matplotlib.pyplot as plt
import os
from datetime import datetime
class LivePlot():

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Episodes x 10")
        self.ax.set_ylabel("Returns")
        self.ax.set_title("Returns over Episodes")

        self.data = None
        self.eps_data = None

        self.epochs = 0

    def update_plot(self, stats):
        if 'AvgReturns' in stats and 'EpsilonCheckpoint' in stats:
            self.data = stats['AvgReturns']
            self.eps_data = stats['EpsilonCheckpoint']

            self.epochs = len(self.data)

            self.ax.clear()
            self.ax.set_xlim([0, self.epochs])
            self.ax.set_ylim([min(self.data + self.eps_data), max(self.data + self.eps_data)])
            self.ax.grid(True)

            # Use markers and different line styles
            self.ax.plot(self.data, 'b-o', label="Returns", markersize=4)
            self.ax.plot(self.eps_data, 'r-s', label="Epsilon", markersize=4)

            self.ax.set_xlabel("Episodes x 10")
            self.ax.set_ylabel("Returns / Epsilon Value")
            self.ax.set_title("Returns and Epsilon over Episodes")

            self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            if not os.path.exists('plots'):
                os.makedirs('plots')

            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Save as PDF for vectorized output
            self.fig.savefig(f'plots/plot_{current_time}.pdf', bbox_inches='tight')


