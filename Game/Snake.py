import pygame,random,copy,sys, math
from collections import deque
from Game.sittings import *
EMPTY=0
SNAKE=1
SNAKEHEAD=2
APPLE=3
class Snake:
    def __init__(self):
        pygame.init()
        self.window=pygame.display.set_mode((WIDTH,HEIGHT))
        self.running = True
        self.win=False
        self.grid= [[EMPTY for _ in range(ROWS)] for _ in range(COLS)]
        self.state = "playing"
        self.snakeCells=deque(maxlen=5)
        self.snakeCells.append((random.randint(0,COLS-1),random.randint(0,ROWS-1)))
        self.snakeCells.append((self.snakeCells[-1][0]+1,self.snakeCells[-1][1]))
        self.snakeCells.append((self.snakeCells[-1][0]+1,self.snakeCells[-1][1]))
        self.snakeCells.append((self.snakeCells[-1][0]+1,self.snakeCells[-1][1]))
        self.snakeCells.append((self.snakeCells[-1][0]+1,self.snakeCells[-1][1]))
        snakeX=random.randint(0,COLS-1)
        snakeY=random.randint(0,ROWS-1)
        self.grid[snakeX][snakeY]=SNAKEHEAD
        self.snakeCells.append((snakeX,snakeY))
        self.score=0
        self.font = pygame.font.SysFont('arial',int(cell_size*2))
        appleX=random.randint(0,COLS-1)
        appleY=random.randint(0,ROWS-1)
        self.applePos=(appleX,appleY)
        self.grid[appleX][appleY]=APPLE
        self.moveVector=[1,0]
        self.updateGrid()

#main game functions
    #starts new game
    def start_new_game(self):
        self.running = True
        self.grid= [[EMPTY for _ in range(ROWS)] for _ in range(COLS)]
        self.state = "playing"
        self.snakeCells=deque(maxlen=2)
        self.snakeCells.append((random.randint(0,COLS-1),random.randint(0,ROWS-1)))
        self.snakeCells.append((self.snakeCells[-1][0]+1,self.snakeCells[-1][1]))
        self.score=0
        self.applePos=(
            random.randint(0,COLS-1),
            random.randint(0,ROWS-1),
        )
        self.moveVector=[1,0]
        self.updateGrid()

    #main loop of the game
    def run(self):
        while self.running:
            self.events()
            self.update()
            self.draw()
            pygame.time.delay(50)
            
        pygame.quit()
        sys.exit()
    
    #1 step of the game for the Ai
    def step(self):
        self.draw()
        self.update()
        self.events()
        # pygame.time.delay(50)

    
    #handles all kinds of ingame events
    def events(self):
        for event in pygame.event.get():
            if event.type== pygame.QUIT:
                self.running=False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if(event.unicode=='a' or event.unicode=='A'):
                    #move left
                    self.moveVector=[-1,0]
                if(event.unicode=='s' or event.unicode=='S'):
                    #move down
                    self.moveVector=[0,1]
                if(event.unicode=='d' or event.unicode=='D'):
                    #move right
                    self.moveVector=[1,0]
                if(event.unicode=='w' or event.unicode=='W'):
                    #move up
                    self.moveVector=[0,-1]
    
    #main draw function
    def draw(self):
        self.window.fill(BLACK)
             
        self.shadeSnakeCells(self.window,self.snakeCells,GREEN)
        self.shadeSnakeCells(self.window,[self.snakeCells[-1]],DARK_GREEN)
        

        applePos=(
            self.applePos[0]*cell_size + grid_pos[0]+cell_size//2,
            self.applePos[1]*cell_size+ grid_pos[1]+cell_size//2
            )
        pygame.draw.circle(self.window, RED, applePos, cell_size//2)
        
        pygame.draw.rect(self.window,GRAY,grid_pos,2)
        # self.drawGrid(self.window)
        

        self.textToScreen(self.window,f"Score: {self.score}", (450,grid_pos[1]-20), colour=WHITE)

        pygame.display.update()

    #updates snake movements and reward system
    def update(self):
        new_snake_cell=(
            self.snakeCells[-1][0]+self.moveVector[0],
            self.snakeCells[-1][1]+self.moveVector[1]
        )
        if(new_snake_cell==self.applePos):
            new_snake=deque(maxlen=len(self.snakeCells)+1)
            self.score+=5
            for cell in self.snakeCells:
                new_snake.append(cell)
            self.snakeCells=new_snake
            if(len(self.snakeCells)>=ROWS*COLS):
                print('game over you win!')
                self.running=False
                self.win=True
            self.applePos=(
                random.randint(0,COLS-1),
                random.randint(0,ROWS-1),
            )
            while(self.applePos in self.snakeCells):
                self.applePos=(
                    random.randint(0,COLS-1),
                    random.randint(0,ROWS-1),
                )
        elif(new_snake_cell in self.snakeCells):
            print('game over you lost!')
            self.running=False
        elif(new_snake_cell[0]<0 or new_snake_cell[0]>=COLS or new_snake_cell[1]<0 or new_snake_cell[1]>=ROWS):
            print('Game Over you lost')
            self.running=False
        self.snakeCells.append(new_snake_cell)   
     
        self.updateGrid()

#update helpers
    def updateGrid(self): 
        self.grid= [[EMPTY for _ in range(ROWS)] for _ in range(COLS)]
        self.grid[self.applePos[0]][self.applePos[1]]=APPLE
        for i in range(len(self.snakeCells)-1):
            try:
                snakePos= self.snakeCells[i]
                self.grid[snakePos[0]][snakePos[1]]=SNAKE
            except:
                pass
                # print('out of bounds')
        try:
            self.grid[self.snakeCells[-1][0]][self.snakeCells[-1][1]]=SNAKEHEAD
        except:
            pass
            # print('out of bounds')

#draw helpers
    #draws the snake grid
    def drawGrid(self,window):
        for r in range(ROWS):
            start_x=grid_pos[0]
            start_y= grid_pos[1]+(r*cell_size)
            end_x=grid_pos[0]+grid_pos[2]
            end_y= grid_pos[1]+(r*cell_size)
            pygame.draw.line(window, GRAY,(start_x,start_y),(end_x,end_y),1)
            for c in range(COLS):
                start_x=grid_pos[0]+(c*cell_size)
                start_y= grid_pos[1]
                end_x=grid_pos[0]+(c*cell_size)
                end_y= grid_pos[1]+grid_pos[3]
                pygame.draw.line(window, GRAY,(start_x,start_y),(end_x,end_y),1)  
    
    #shades the snake cells
    def shadeSnakeCells(self,window,cells,color):
            margin=3
            for cell in cells:
                pygame.draw.rect(window,color,(cell[0]*cell_size + grid_pos[0],cell[1]*cell_size+ grid_pos[1],cell_size+margin,cell_size+margin))

    #adds text to the GUI
    def textToScreen(self,window,text, pos, colour=BLACK):
        font = self.font.render(text,False,colour)
        window.blit(font,pos)  

#AI helpers
    def valid(self, action):
        actions = {1:[]}

#general Helpers
    # proper printing function
    def displayGrid(self):
        for col in self.grid:
            for row in col:
                print(f"{row}, ",end='')
            print()