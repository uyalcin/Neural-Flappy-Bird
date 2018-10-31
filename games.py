from tkinter import *
import time
import random

# Parameters
size_window_x = 1024
size_window_y = 768

birdSize = 30
initPos = (300, size_window_y/2 - birdSize/2)

wallSpeed = 200
intervallWalls = 1.5
wallWidth = 50
wallHeight = size_window_y
wallSpacement = 200

G = 9.8
gameSpeed = 3.0

fenetre = Tk()
fenetre.title("Neural flappy")
canvas = Canvas(fenetre, width=size_window_x, height=size_window_y)
canvas.configure(background="white")

birds = []
# Population parameters
n_population = 1

# Creation of the bird
for k in range(n_population):
    birds += [canvas.create_rectangle(initPos[0], initPos[1], initPos[0] + birdSize, initPos[1] + birdSize, fill='black')]
bird = birds[0]

walls = []
validWall = []
last_time = time.time()
t = [0] * n_population
initSecond = 2
seconds = initSecond
last_time_wall = 0

score = 0
lost = False
jump = [True] * n_population

isAI = True

if(isAI):
    from model import *

def jumping(n_bird):
    global t
    t[n_bird] = 0
    global jump
    jump[n_bird] = True

def motion(event):
    jumping(t[0])

fenetre.bind('<Button-1>', motion)
fenetre.bind('<space>', motion)

scoreText = canvas.create_text(size_window_x / 2, 80, text="0", anchor=CENTER, font=('Helvetica', '50', "bold"))

canvas.pack()

# Write to a file the features
file = open("data", "a+")

if(isAI):
    # Creation of the population
    neurals = []
    for k in range(n_population):
        neurals += [NN()]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

frame = 0

while True:
    deltaTime = (time.time() - last_time) * gameSpeed
   
    for i, bird in enumerate(birds):
        dY = -(3 - t[i]**2) * deltaTime * 100
        canvas.move(bird, 0, dY)
        canvas.move(bird, 0, (G * t[i]**2) * deltaTime * 200)

        t[i] += deltaTime % 60
    seconds += deltaTime % 60

    # Instantiate the walls
    if(seconds - last_time_wall >= intervallWalls):
        yPosition = random.randint(100, size_window_y - wallSpacement - 100)

        walls += [canvas.create_rectangle(0, 0, wallWidth, yPosition, fill="black")]
        canvas.move(walls[-1], size_window_x, 0)

        walls += [canvas.create_rectangle(0, 0, wallWidth, wallHeight, fill="black")]
        canvas.move(walls[-1], size_window_x, yPosition + wallSpacement)

        validWall += [False]
        last_time_wall = seconds

    # Move the walls
    for wall in walls:
        canvas.move(wall, -deltaTime * wallSpeed, 0)

    # Collision
    posBird = canvas.coords(bird)

    for wall in walls:
        posWall = canvas.coords(wall)
        if(posBird[2] >= posWall[0] and
                posBird[0] <= posWall[2] and
                posBird[3] >= posWall[1] and
                posBird[1] <= posWall[3]):
            lost = True

    hideWalls = []
    for i in range(len(walls)):
        posWall = canvas.coords(walls[i])
        if(posWall[2] < 0):
            canvas.delete(walls[i])
            hideWalls += [i]

    for i in range(len(hideWalls)):
        del(walls[0])
    for i in range(len(hideWalls) // 2):
        del(validWall[0])

    # If the player is out of the window
    if(posBird[1] > size_window_y or posBird[3] < 0):
        lost = True

    # Score
    for i in range(0, len(walls), 2):
        idWall = i // 2
        posWall = canvas.coords(walls[i])
        if(posBird[0] > posWall[2] and not validWall[idWall]):
            score += 1
            validWall[idWall] = True
    
    # Features
    features_1 = 0
    features_2 = 0
    for i in range(0, len(walls), 2):
        idWall = i // 2
        posWall = canvas.coords(walls[i])
        if not validWall[idWall]:
            features_1 = posWall[2] - posBird[0]
            features_2 = (-1) * ((posBird[1] + posBird[3]) / 2.0 - (posWall[3] + wallSpacement / 2.0)) 
            break
    if(isAI):    
        if(frame % 50 == 0):
            for i in range(n_population):
                pred = sess.run(tf.argmax(neurals[i].a, 1), feed_dict={x: [[features_1, features_2]]})
                #print(pred)
                if(pred == 1):
                    jumping(i)
    else:
        file.write(str(features_1) + "/" + str(features_2) + "/" + str(float(jump)) + "\n")

    for i in jump:
        if(i):
            #print("jump")
            i = False


    # Display score
    canvas.itemconfigure(scoreText, text=str(score), fill='#445b75')
    canvas.tag_raise(scoreText)

    # Reset the game if lost
    if(lost):
        score = 0
        for wall in walls:
            canvas.delete(wall)
        walls = []
        validWall = []
        for bird in birds:
            canvas.coords(bird, initPos[0], initPos[1], initPos[0] + birdSize, initPos[1] + birdSize)
        
        t = [0] * n_population
        second = initSecond
        last_time_wall = 0
        lost = False

    frame += 1
    last_time = time.time()
    canvas.update_idletasks()
    canvas.update()

file.close()