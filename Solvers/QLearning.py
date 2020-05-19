import numpy as np
import copy, random,math,time
class QL:
    def __init__(self,game):
        self.game=game
        self.num_cols=len(self.game.grid)
        self.num_rows=len(self.game.grid[0])
        self.observation_space_size = [2]*9+[2,2]
        self.action_space=4
        self.q_table=np.random.uniform(low=-2,high=0,size=(self.observation_space_size+[self.action_space]))

    def reset(self):
        self.game.start_new_game()
        return self.get_observation()
    
    def get_observation(self):
        observations = [0]*len(self.observation_space_size)
        snake_head = self.game.snakeCells[-1]
        points_to_obs =[
            [-1,-1],[0,-1],[1,-1],
            [-1,0],[0,0],[1,0],
            [-1,1],[0,1],[1,1],
        ]
        for i in range(len(self.observation_space_size)-2):
            point = points_to_obs[i]
            try:
                if(self.game.grid[snake_head[0]+point[0]][snake_head[1]+point[1]] == 0):
                    observations[i] = 0
                else:
                    observations[i] = 1
            except(IndexError):
                observations[i]= 1 #wall or out of bounds
        if((snake_head[0]-self.game.applePos[0])>0):
            positiveX = 1
        else:
            positiveX=0
        if((snake_head[1]-self.game.applePos[1])>0):
            positiveY = 1
        else:
            positiveY=0
         
        observations[-2] = positiveX
        observations[-1] = positiveY
        return tuple(observations)
            

    def step(self,action):
        actions = {0:[0,-1],1:[0,1],2:[-1,0],3:[1,0]}
                #up,down,left,right
        prev_score=self.game.score
        distance_before = math.sqrt((self.game.snakeCells[-1][0]-self.game.applePos[0])**2+(self.game.snakeCells[-1][1]-self.game.applePos[1])**2)
        self.game.moveVector = actions[action]
        distance_after = math.sqrt((self.game.snakeCells[-1][0]-self.game.applePos[0])**2+(self.game.snakeCells[-1][1]-self.game.applePos[1])**2)
        new_score = self.game.score
        done = not self.game.running
        if(done):
            if(self.game.win):
                reward = 100
                new_observation =[]
            else:
                print('lost')
                reward = -100
                new_observation = []
        else:
            if(new_score>prev_score):
                reward = 50
            elif(distance_after>distance_before):
                reward = 40
            else:
                reward = 20
            new_observation=self.get_observation()
        self.game.step()
        return new_observation,reward,done

    def solve(self):
        total_games=100000
    
        learning_rate=0.1
        discount = 0.90
        
        epsilon=0.9
        start_decay=0
        end_decay=total_games//3
        
        epsilon_decay_value=epsilon/(end_decay-start_decay)

        for episode in range(total_games):
            observation = self.reset()
            print(f"game number {episode}")
            done = False
            while not done:
                if(np.random.random()>epsilon):
                    action = np.argmax(self.q_table[observation])
                else:
                    action = random.randint(1,self.action_space-1)
                
                
                new_observation, reward, done = self.step(action)
               
                if not done:
                    max_future_q=np.max(self.q_table[new_observation])
                    current_q=self.q_table[observation+(action,)] 
                    
                    new_q = (1-learning_rate) * current_q + learning_rate*(reward + discount*max_future_q)
                    self.q_table[observation+(action,)]=new_q

                elif(reward ==10 ):
                    self.q_table[observation+(action,)]=10
                    print("made it!!")

                observation = new_observation
            if(end_decay>=episode >=start_decay):
                epsilon-=epsilon_decay_value
        
        self.final_sol()
  
    def final_sol(self):
        observation = self.reset()
        while self.game.running:
            action = np.argmax(self.q_table[observation])
            observation, reward, done=self.step(action)
            time.sleep(0.1)
            if(done):
                break
        
