import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from collections import deque
import numpy as np
import random,copy,math,time

REPLAY_MEMORY_SIZE= 1000
BATCH_SIZE= 128
UPDATE_TARGET_EVERY=10
DISCOUNT=0.9
class DQNAgent:
    def __init__(self,game):
        self.game=game

        # self.observation_space = (80,60,1)
        self.observation_space = (12,)

        self.action_space = 4

        #main  get trained evert batch
        # self.model=self.createModel()
        self.model = keras.models.load_model('Solvers/models/target_model.h5')  


        #target predicts every step
        # self.targetModel= self.createModel()
        self.targetModel=keras.models.load_model('Solvers/models/target_model.h5') 

        # self.targetModel.set_weights(self.model.get_weights())
        
        self.replayMemory= deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.targetUpdateCounter=0

        self.lastAte=0

        
 
    #creates model for training and predicting
    def createModel(self):
        model = Sequential()
        
        # model.add(Conv2D(16,(3,3), input_shape=self.observation_space))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(2,2))
        # model.add(Dropout(0.3))
        
        # model.add(Conv2D(16,(3,3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(2,2))
        # model.add(Dropout(0.3))
        
        # model.add(Flatten())
        # model.add(Dense(32,activation='relu'))

        model.add(Dense(units=64,input_shape=self.observation_space,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(32,activation='relu'))   
        model.add(Dense(32,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(32,activation='relu'))   
        model.add(Dense(32,activation='relu'))       
        model.add(Dense(self.action_space, activation="linear"))
        model.compile(loss="mse", optimizer='rmsprop', metrics=['accuracy'])
        return model
    
    #adds transitions to replay memory for training later
    def updateReplyMemory(self, trainsition):
        self.replayMemory.append(trainsition)
    
    #getting actions
    def getQs(self, observation, epsilon):
        if(np.random.random()>epsilon):
            return self.targetModel.predict(observation.reshape((1,*self.observation_space)))[0]
        else:
            rand =random.randint(1,self.action_space-1)
            return rand
            
    #main train function
    def train(self,terminal_state):
        if(len(self.replayMemory) < REPLAY_MEMORY_SIZE):
            return
        
        minibatch= random.sample(self.replayMemory, BATCH_SIZE)
        
        observations = np.array([transition[0] for transition in minibatch])

        currentQsList= self.targetModel.predict(observations)
        
        newObservations= np.array([transition[3] for transition in minibatch])

        futureQsList = self.targetModel.predict(newObservations)

        x=[]
        y=[]
        for index,(observation, action, reward,newObservation,done) in enumerate(minibatch):
            if not done:
                maxFututeQ= np.max(futureQsList[index])
                newQ= reward+ DISCOUNT * maxFututeQ
            else:
                newQ= reward
            
            currentQs= currentQsList[index]
            currentQs[action]= newQ
            
            x.append(observation)
            y.append(currentQs)
        
        self.model.fit(np.array(x), np.array(y),batch_size=BATCH_SIZE,verbose=0,shuffle=False)
        
        if terminal_state:
            self.targetUpdateCounter+=1
            
            
        if self.targetUpdateCounter>UPDATE_TARGET_EVERY:
            print('target updated!')
            self.targetModel.set_weights(self.model.get_weights())
            self.targetUpdateCounter=0
            self.targetModel.save('Solvers/models/target_model.h5')


    def reset(self):
        self.game.start_new_game()
        self.game.step()
        self.game.step()
        self.lastAte=0    
        return self.get_observation()
    
    def get_observation(self):
        # return np.array(self.game.grid).reshape(self.observation_space)
        observations = [0]*(self.observation_space[0])
        snake_head = self.game.snakeCells[-1]
        points_to_obs =[
            [-1,-1],[0,-1],[1,-1],
            [-1,0],[0,0],[1,0],
            [-1,1],[0,1],[1,1],
        ]
        for i in range((self.observation_space[0])-3):
            point = points_to_obs[i]
            try:
               self.game.grid[snake_head[0]+point[0]][snake_head[1]+point[1]]
            except(IndexError):
                observations[i]= 4 #wall or out of bounds
        
        distanceX=(snake_head[0]-self.game.applePos[0])
        distanceY=(snake_head[1]-self.game.applePos[1])
        ecludean = math.sqrt((snake_head[0]-self.game.applePos[0])**2+(snake_head[1]-self.game.applePos[1])**2)
         
        observations[-3] = distanceX
        observations[-2] = distanceY
        observations[-1] = ecludean
        return np.array(observations)

            
    def step(self,action):
        actions = {0:[0,-1],1:[0,1],2:[-1,0],3:[1,0]}
                #up,down,left,right
        prev_score=self.game.score
        distance_before = math.sqrt((self.game.snakeCells[-1][0]-self.game.applePos[0])**2+(self.game.snakeCells[-1][1]-self.game.applePos[1])**2)
        self.game.moveVector = actions[action]
        self.game.step()
        distance_after = math.sqrt((self.game.snakeCells[-1][0]-self.game.applePos[0])**2+(self.game.snakeCells[-1][1]-self.game.applePos[1])**2)
        new_score = self.game.score
        done = not self.game.running
        if(done):
            if(self.game.win):
                reward = 100
            else:
                print('lost')
                reward = -100
        else:
            if(new_score>prev_score):
                reward = 100
                self.lastAte=0
            else:
                if(self.game.score<50):
                    if(distance_after<distance_before):
                        reward = 100-self.game.score
                    else:
                        reward = -40
                    
                    if(self.lastAte >=160):
                        reward = -40
                        self.lastAte =0
                        self.game.running=False
                        done = True
                else:
                    reward = 0


        new_observation=self.get_observation()
        self.lastAte+=1
        return new_observation,reward,done
    
    def solve(self):
        total_games=150_000
        
        epsilon=0.5
        start_decay=0
        end_decay=total_games//4
        
        epsilon_decay_value=epsilon/(end_decay-start_decay)

        for episode in range(total_games):
            observation = self.reset()
            print(f"game number {episode}")
            done = False
            while not done:

                qValues=self.getQs(observation, epsilon)
                
                action = np.argmax(qValues)
                
                new_observation, reward, done = self.step(action)

                self.updateReplyMemory(
                    (observation, action, reward, new_observation, done)
                )

                observation = new_observation
                self.train(done)    
            if(end_decay>=episode >=start_decay):
                epsilon *=epsilon_decay_value
            print('score for this episode is ',self.game.score)
        
        self.model.save('Solvers/models/DQN.h5')
        self.final_sol()
    
    def final_sol(self):
        observation = self.reset()
        while self.game.running:
            qValues=self.getQs(observation, 0)  
            action = np.argmax(qValues)
            observation, reward, done=self.step(action)
            time.sleep(0.1)
            if(done):
                break
    