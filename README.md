
# Neural Flappy Bird

Neural Flappy Bird is a simple artificial intelligence which play to the famous game **Flappy Bird**. The AI is based on a **neural network** and a **genetic algorithm**. 

## Neural Network
Here is the architecture of the neural network :
![enter image description here](https://raw.githubusercontent.com/uyalcin/Neural-Flappy-Bird/master/screenshots/nn.png)

Basically, the input layer have 2 neurons : the horizontal distance to the closest wall and the height difference to the closest gap. The features are represented here :

![enter image description here](https://raw.githubusercontent.com/uyalcin/Neural-Flappy-Bird/master/screenshots/1.png)
Then there is a hidden layer with 10 neurons (with ReLU activation), and finally return the decision of the bird : jump or not jump.
The code of the neural network is implemented in the file **neural_network.py** with the numpy library. 

## Genetic algorithm

In each generation, with send a population of 10 birds with a random neural network. For each birds, we calculate his fitness function to determine a ranking of the best birds. The fitness function is simply the distance traveled by the bird. We take the 5 best birds and pass them directly to the next generation, and we remove the 5 weakest birds. Finally, we make random mutations of the 5 best birds, that means we add or remove random values to the weights of the neural network.

The mutation algorithm is detailed in the file **model.py**.

## Results

By doing so, we can have a great bird level, after several tens of generations.
If you want to test the level of the AI, just do the following command in the root of the project :

    python3 games.py

You can adjust the different parameters of the games, for example you can try to change the wall spacing,
or the game speed. Enjoy !

