
# Neural Flappy Bird

Neural Flappy Bird is a simple artificial intelligence which play to the famous game **Flappy Bird**. The AI is based on a **neural network** and a **genetic algorithm**. 
![enter image description here](https://raw.githubusercontent.com/uyalcin/Neural-Flappy-Bird/master/screenshots/neural_flappy.png)

## Neural Network
Here is the architecture of the neural network :
![enter image description here](https://raw.githubusercontent.com/uyalcin/Neural-Flappy-Bird/master/screenshots/nn.png)

Basically, the input layer has 2 neurons : the horizontal and vertical distances to the closest wall. The features are represented here :

![enter image description here](https://raw.githubusercontent.com/uyalcin/Neural-Flappy-Bird/master/screenshots/1.png)
Then there is a hidden layer with 10 neurons (with ReLU activation), and finally return the decision of the bird : to jump or not to jump.
The code of the neural network is implemented in the file **neural_network.py** with the numpy library. 

## Genetic algorithm

In each generation, we send a population of 10 birds with a random neural network. For each bird, we calculate its fitness function to determine a ranking of the best birds. The fitness function is simply the distance traveled by the bird. We take the 5 best birds and pass them directly to the next generation, and we remove the 5 worst birds. Finally, we make random mutation of the 5 best birds, meaning we add or remove random values to the weights of the neural network.

The mutation algorithm is detailed in the file **model.py**.

## Results

By doing so, we can have a great bird level, after several dozens generations.
If you want to test the level of the AI, just execute the following command in the root of the project :

    python3 games.py

You can adjust the different parameters of the games, for example you can try to change the wall spacing,
or the game speed. Enjoy !

