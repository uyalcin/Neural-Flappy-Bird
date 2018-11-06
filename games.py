from tkinter import *
import time
import random
import tensorflow as tf
from util import *
from neural_network import NeuralNetwork

# Parameters
size_window_x = 1024
size_window_y = 768

birdSize = 30
initPos = (300, size_window_y/2 - birdSize/2)

wallSpeed = 200
intervallWalls = 1.5
wallWidth = 50
wallHeight = size_window_y
wallSpacement = 150

G = 9.8
gameSpeed = 1.5

fenetre = Tk()
fenetre.title("Neural flappy")
canvas = Canvas(fenetre, width=size_window_x, height=size_window_y)
canvas.configure(background="white")

birds = []
# Population parameters
n_population = 10

def initBirds():
    global birds
    birds = []
    # Creation of the birds
    for k in range(n_population):
        birds += [canvas.create_rectangle(initPos[0], initPos[1], initPos[0] + birdSize, initPos[1] + birdSize, fill='black')]

initBirds()
walls = []
validWall = []
last_time = time.time()
t = [0] * n_population
initSecond = 2
seconds = initSecond
last_time_wall = 0
generation = 1

scores = [0] * n_population
lost = [False] * n_population
jump = [True] * n_population
fitness = [0] * n_population

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
generationText = canvas.create_text(50, 50, text="Generation : 1", anchor=W, font=('Helvetica', '20', "bold"))

canvas.pack()

# Write to a file the features
#file = open("data", "a+")

if(isAI):
    # Creation of the population
    neurals = []
    for k in range(n_population):
        neurals += [NeuralNetwork(N_features, N_neurals, N_classes)]
    #sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    #predictions = [tf.argmax(neurals[i].a, 1) for i in range(n_population)]
    #sess.graph.finalize()

frame = 0
features_1 = [0] * n_population
features_2 = [0] * n_population


while True:
    deltaTime = (time.time() - last_time) * gameSpeed

    for i, bird in enumerate(birds):

        # If the current bird is dead, pass
        if(bird == None):
            continue

        dY = -(3 - t[i]**2) * deltaTime * 100
        canvas.move(bird, 0, dY)
        canvas.move(bird, 0, (G * t[i]**2) * deltaTime * 200)

        t[i] += deltaTime % 60

        # Collision
        posBird = canvas.coords(bird)

        for wall in walls:
            posWall = canvas.coords(wall)
            if(posBird[2] >= posWall[0] and
                    posBird[0] <= posWall[2] and
                    posBird[3] >= posWall[1] and
                    posBird[1] <= posWall[3]):
                lost[i] = True

        # If the player is out of the window
        if(posBird[1] > size_window_y or posBird[3] < 0):
            lost[i] = True

        # Score
        for j in range(0, len(walls), 2):
            idWall = j // 2
            posWall = canvas.coords(walls[j])
            if(posBird[0] > posWall[2] and not validWall[idWall]):
                scores[i] += 1
                validWall[idWall] = True

        # Features
        for j in range(0, len(walls), 2):
            idWall = j // 2
            posWall = canvas.coords(walls[j])
            if not validWall[idWall]:
                features_1[i] = posWall[2] - posBird[0]
                features_2[i] = (-1) * ((posBird[1] + posBird[3]) / 2.0 - (posWall[3] + wallSpacement / 2.0))
                break

        # Fitness function for each bird
        fitness[i] += deltaTime * wallSpeed


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


    # Make predictions
    if(isAI):
        if(frame % 50 == 0):
            for i in range(n_population):
                #pred = sess.run(predictions[i], feed_dict={x: [[features_1[i], features_2[i]]]})
                pred = neurals[i].predict([features_1[i], features_2[i]])
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
    canvas.itemconfigure(scoreText, text=str(max(scores)) + ", In Life : " + str(len([k for k in birds if k!=None])), fill='#445b75')
    canvas.tag_raise(scoreText)

    # Display generation
    canvas.itemconfigure(generationText, text="Generation : " + str(generation), fill='#ff0000')
    canvas.tag_raise(generationText)


    # Reset the game if lost
    for i, l in enumerate(lost):
        if l:
            canvas.delete(birds[i])
            birds[i] = None

    if(not(False in lost)):
        scores = [0] * n_population
        for wall in walls:
            canvas.delete(wall)
        walls = []
        validWall = []
        t = [0] * n_population
        second = initSecond
        last_time_wall = 0

        initBirds()

        # Get the indices of the best birds
        #print(fitness)
        ranking = sorted(range(n_population), key=lambda x: fitness[x])[::-1]
        #print(ranking)
        bestBirds = ranking[:5]

        nextNN = []
        for ibird in bestBirds:
            nextNN += [neurals[ibird]]
            #print(nextNN[0].weights1)
            mut = mutateNN(neurals[ibird])
            #print(nextNN[0].weights1)
            nextNN += [mut]
            print(nextNN[-2] == nextNN[-1])

        neurals = list(nextNN)
        for k in nextNN:
            print(k.weights1)

        #predictions = [tf.argmax(neurals[i].a, 1) for i in range(n_population)]

        # Init each variables
        scores = [0] * n_population
        lost = [False] * n_population
        jump = [True] * n_population
        fitness = [0] * n_population
        generation += 1

    frame += 1
    last_time = time.time()
    canvas.update_idletasks()
    canvas.update()

file.close()
