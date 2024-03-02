from game import MineSweeper
from renderer import Render
import numpy as np 
import time
import pygame
import matplotlib.pyplot as plt
import random
import sys
import json
import pprint
q_table = [
    # #   0  1  2
    # #   TL F TR
    #     [2,0,0], # Left Close  0
    #     [0,1,0], # Left Med    1
    #     [2,1,2], # Left Far    2
    #     [3,0,0], # Front Close 3
    #     [1,1,0], # Front Med   4
    #     [0,1,0], # Front Far   5
    #     [2,2,0], # Right Close 6
    #     [0,1,0], # Right Med   7
    #     [1,0,2]  # Right Far   8
            [0.01 for i in range(20)] for j in range(20)
            ]
q_table = np.array(q_table)
# print(q_table)
class Play():
    def __init__(self):
        self.width = 20
        self.height = 20
        self.bombs = 20
        self.env = MineSweeper(self.width,self.height,self.bombs)
        self.renderer = Render(self.env.state)
        self.renderer.state = self.env.state

    def do_step(self,i,j):
        # i=int(i/30)
        # j=int(j/30)
        next_state,terminal,reward = self.env.choose(i,j)
        self.renderer.state = self.env.state
        self.renderer.draw()
        return next_state,terminal,reward
def weightedChoice(elements, weights):
    # verify the lists are the same length
    assert len(elements) == len(weights)
    total = sum(weights)
    r = random.uniform(0, total)
    w = 0
    for i in range(len(elements)):
        w += weights[i]
        if w > r:
            return elements[i]
    # all weights are zero if we get here, so pick at random
    return random.choice(elements)

def main():
    
    play = Play()
    play.renderer.draw()
    print(play.env.grid)
    moves = []
    epsilon = .90
    gamma = .99
    lr = .1
    XEpisodes = []
    score = 0
    games = 1
    wins = 0
    losses = 0
    YOutcomes = []
    avgSquares = 0
    # fig = plt.pl
    for episode in range(1000):    
        moves = []            
        for l in range(250):
            ###Learning code

            if random.uniform(0,1) > epsilon or len(sys.argv) > 1: # Greedy choice
                print("Greedy Choice")
                xy = weightedChoice([i for i in range(len(q_table.flatten()))], q_table.flatten())
                print("Weighted Choice: "+str(xy))
                index=[int(xy/play.height),(xy%play.height)]
                print(index)
                x = index[0]
                y = index[1]
                while (x,y) in moves:
                    # print("new choice")
                    q_table[x][y] = round((1-lr) * q_table[x][y] + lr * (-5 + gamma) * max(q_table[x][:]),4)
                    xy = weightedChoice([i for i in range(len(q_table.flatten()))], q_table.flatten())
                    index=[int(xy/play.height),(xy%play.height)]
                    x = index[0]
                    y = index[1]
                moves.append((x,y))
                print("Coords: ("+str(x)+", "+str(y)+")")
                currentBoard,terminal,reward= play.do_step(x,y)
                # print("Current State:\n"+str(currentBoard))
                print("Picked: "+str(currentBoard[x][y]))
                print("Reward: "+str(reward))
                print()
                score += reward
                q_table[x][y] = round((1-lr) * q_table[x][y] + lr * (reward + gamma) * max(q_table[x][:]),4)
                if(terminal):
                    if(reward==-1):
                        print("LOSS")
                        losses += 1
                    else: 
                        print("WIN")
                        wins += 1
                    moves = []
                    # YOutcomes.append(score)
                    # YOutcomes.append(play.env.uncovered_count)
                    avgSquares += play.env.uncovered_count
                    # XEpisodes.append(games)
                    # score = 0
                    games += 1

                    q_table[x][y] = round((1-lr) * q_table[x][y] + lr * (reward + gamma) * max(q_table[x][:]),4)
                    # print('\n'.join(str(row) for row in q_table))
                    print(q_table)
                    play.env.reset()
                    play.renderer.state=play.env.state
                    play.renderer.draw()
                    print(play.env.grid)
                    # YOutcomes.append(wins/losses)
            else: # Random choice
                # print("Random Choice")
                y = random.randint(0,play.height-1)
                x = random.randint(0,play.width-1)
                while (x,y) in moves:
                    # print("new choice")
                    y = random.randint(0,play.height-1)
                    x = random.randint(0,play.width-1)
                moves.append((x,y))
                # print("Coords: ("+str(x)+", "+str(y)+")")
                currentBoard,terminal,reward= play.do_step(x,y)
                # print("Current State:\n"+str(currentBoard))
                # print("Picked: "+str(currentBoard[x][y]))
                # print("Reward: "+str(reward))
                # print(play.env.uncovered_count)
                # print()
                q_table[x][y] = round((1-lr) * q_table[x][y] + lr * (reward + gamma) * max(q_table[x][:]),4)
                if(terminal):
                    if(reward==-1):
                        print("LOSS")
                        losses += 1
                    else: 
                        print("WIN")
                        wins += 1
                    moves = []
                    q_table[x][y] = round((1-lr) * q_table[x][y] + lr * (reward + gamma) * max(q_table[x][:]),4)
                    # print('\n'.join(str(row) for row in q_table))
                    print(q_table)
                    moves = []
                    # YOutcomes.append(score)
                    # YOutcomes.append(play.env.uncovered_count)
                    avgSquares += play.env.uncovered_count
                    # XEpisodes.append(games)
                    # score = 0
                    games += 1
                    play.env.reset()
                    play.renderer.state=play.env.state
                    play.renderer.draw()
                    print(play.env.grid)
        # YOutcomes.append(score/games)
        YOutcomes.append(wins/games)
        # YOutcomes.append(avgSquares/games)
        # score = 0
        # games = 0
        XEpisodes.append(episode)
        plt.plot(XEpisodes,YOutcomes)
        epsilon -= .0008
        plt.savefig('graph.png')
    print(q_table)
    with open('output.txt', 'w') as filehandle:
        json.dump(list(q_table), filehandle)
    # while(True):
        ##Testing code
        # y = random.randint(0,play.height-1)
        # x = random.randint(0,play.width-1)
        # while (x,y) in moves:
        #     # print("new choice")
        #     y = random.randint(0,play.height-1)
        #     x = random.randint(0,play.width-1)
        # xy = weightedChoice([i for i in range(len(q_table.flatten()))], q_table.flatten())
        # # index= np.unravel_index(xy.item(), currentBoard.shape)
        # index=[int(xy/play.height),(xy%play.height)]
        # print("Weighted Choice: "+str(index))
        # moves.append((x,y))
        # # print(moves)
        # print("Coords: ("+str(x)+", "+str(y)+")")
        # currentBoard,terminal,reward= play.do_step(x,y)
        # # print("Current State:\n"+str(currentBoard))
        # print("Picked: "+str(currentBoard[x][y]))
        # print("Reward: "+str(reward))
        # # print(play.env.uncovered_count)
        # print()
        # q_table[x][y] = round((1-lr) * q_table[x][y] + lr * (reward + gamma) * max(q_table[x][:]),4)
        # if(terminal):
        #     if(reward==-1):
        #         print("LOSS")
        #     else: 
        #         print("WIN")
        #     q_table[x][y] = round((1-lr) * q_table[x][y] + lr * (reward + gamma) * max(q_table[x][:]),4)
        #     # print('\n'.join(str(row) for row in q_table))
        #     print(q_table)
        #     play.env.reset()
        #     play.renderer.state=play.env.state
        #     play.renderer.draw()
        #     print(play.env.grid)

    # print(q_table)
    # print(pprint.pformat(q_table))
    # print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in q_table]))
    # print('\n'.join(str(row) for row in q_table))
    # with open('output.txt', 'w') as filehandle:
    #     json.dump(q_table, filehandle)



    # time.sleep(10)
        # events = play.renderer.bugfix()
        # for event in events:
        #     if(event.type==pygame.MOUSEBUTTONDOWN):
        #         y,x = pygame.mouse.get_pos()
        #         # print(y,x)
        #         print("Coords: ("+str(x)+", "+str(y)+")")
        #         currentBoard,terminal,reward= play.do_step(x,y)
        #         print("Current Board: "+str(currentBoard))
        #         print("Reward: "+str(reward))
        #         print("Terminal: "+str(terminal))
        #         if(terminal):
        #             if(reward==-1):
        #                 print("LOSS")
        #             else: 
        #                 print("WIN")
        #             play.env.reset()
        #             play.renderer.state=play.env.state
        #             play.renderer.draw()
        #             print(play.env.grid)
                    
main()