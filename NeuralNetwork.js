class ActivationFunction{
    
    // func; // Fonction d'activation
    // der; // Dérivée de la fonction d'activation

    constructor(func, der){
        this.func = func;
        this.der = der;
    }

    static sigmoid(){
        return new ActivationFunction(
            x => 1 / (1 + Math.exp(-1 * x)),
            y => y * (1 - y)
        );
    }

    static relu(){
        return new ActivationFunction(
            x => x <= 0 ? 0 : x,
            y => y <= 0 ? 0 : 1
        );
    }

    static linear(){
        return new ActivationFunction(
            x => x,
            y => 1
        );
    }
}

class Neuron{

    // weights = 0; // Poids de tous les neurones de la couche précédente vers ce neurone
    // bias; // Bias de ce neurone

    constructor(nbInputs){
        // On initialise les poids avec une valeur entre -1 et 1
        this.weights = [];
        for(let i = 0; i < nbInputs; i++){
            this.weights.push((Math.random() * 2) - 1);
        }
        // On initialise le bias avec une valeur entre -1 et 1
        this.bias = (Math.random() * 2) - 1;
    }

    // Calcule la sortie du neurone
    evaluate(inputs, activationFunction){
        // Somme des valeurs d'entrées * les poids
        let value = 0;
        for(let i = 0; i < this.weights.length; i++){
            value += inputs[i] * this.weights[i];
        }
        // On ajoute le bias
        value += this.bias;
        // On applique la fonction d'activation
        return activationFunction.func(value);
    }
}

class Layer{

    // neurons; // Neurones de cette couche
    // activationFunction; // Fonction d'activation de cette couche

    constructor(layerSize, previousLayerSize, activationFunction){
        // Génération des neurones de cette couche
        this.neurons = [];
        for(let i = 0; i < layerSize; i++){
            this.neurons.push(new Neuron(previousLayerSize));
        }
        // Enregistrement de la fonction d'activation
        this.activationFunction = activationFunction;
    }

    // Evalue le résultat de ses neurones par rapport aux valeurs envoyées de la couche précédente
    evaluate(inputs){
        let outputs = [];
        for(let neuron of this.neurons){
            outputs.push(neuron.evaluate(inputs, this.activationFunction));
        }
        return outputs;
    }
}

class NeuralNetwork{

    // layers; // Couches du réseau de neurones
    // learningRate; // Taux d'apprentissage

    constructor(learningRate = 0.03){
        // Taux d'apprentissage
        this.learningRate = learningRate;
        // Couches
        this.layers = [];
    }

    addLayer(layerSize, activationFunction){
        this.layers.push(
            new Layer(
                layerSize, 
                this.layers.length > 0 ? this.layers[this.layers.length - 1].neurons.length : 0, // Taille de la couche précédente ou 0 si il n'y a pas de couche précédente (couche d'entrée)
                activationFunction
            )
        );
    }

    // Feed-forward (= calcul la/les sortie/s du réseau de neurones sur une/des valeur/s d'entrée) et renvoie les sorties de toutes les couches
    evaluate(inputs){
        let layersOutputs = [inputs];
        for(let i = 1; i < this.layers.length; i++){ // i = 1 car on skip la couche d'entrée, et oui, on a déjà les valeurs (inputs)
            layersOutputs.push(this.layers[i].evaluate(layersOutputs[layersOutputs.length - 1])); // On entraine la couche suivante avec les sorties trouvées l'itération précédente
        }
        return layersOutputs;
    }

    // Backpropagation avec batchs (= apprentissage)
    learn(batchInputs, batchOutputs){
        // Tableau qui va recevoir la somme de toutes les modifications des poids de tous les neurones de toutes les couches
        let weightsModification = [];
        let biasesModification = []; // Idem pour le bias
        // Pour chaque couche du réseau de neurones
        for(let layerIndex = 1; layerIndex < this.layers.length; layerIndex++){
            let arrayWeights1 = [];
            let arrayBiases1 = [];
            // Pour chaque neurone de la couche
            for(let neuronIndex = 0; neuronIndex < this.layers[layerIndex].neurons.length; neuronIndex++){
                let arrayWeights2 = [];
                // Pour chaque poids du neurone
                for(let weightIndex = 0; weightIndex < this.layers[layerIndex].neurons[neuronIndex].weights.length; weightIndex++){
                    arrayWeights2.push(0);
                }
                arrayWeights1.push(arrayWeights2);
                arrayBiases1.push(0);
            }
            weightsModification.push(arrayWeights1);
            biasesModification.push(arrayBiases1);
        }

        // On parcours tous les batchs
        for(let batchIndex = 0; batchIndex < batchInputs.length; batchIndex++){
            // On récupère les entrées/sorties de ce batch
            let inputs = batchInputs[batchIndex];
            let outputs = batchOutputs[batchIndex];

            // Calcul du résultat du réseau de neurones pour ces entrées
            let neuralNetworkOutputs = this.evaluate(inputs);

            // Calcul des gradients des couches
            let gradients = Array(this.layers.length - 1); // - 1 car on ne veut pas la couche d'entrée et c'est un tableau qu'on va remplir en partant de la fin, d'où l'initialisation
            for(let layerIndex = this.layers.length - 1; layerIndex > 0; layerIndex--){ // On commence donc par la dernière couche
                let layerGradients = [];
                // Pour chaque neurone de cette couche
                for(let neuronIndex = 0; neuronIndex < this.layers[layerIndex].neurons.length; neuronIndex++){
                    // Calcul de l'erreur
                    let error = 0
                    if(layerIndex === this.layers.length - 1){ // Si c'est la couche de sortie, on calcul le gradient différemment
                        error = outputs[neuronIndex] - neuralNetworkOutputs[layerIndex][neuronIndex]; // cible - sortie
                    }
                    else{ // Pour le reste des couches, c'est le même calcul (somme des gradients précédents * les poids associés entre ce neurone et les précédents)
                        // Pour chaque neurone de la couche suivante
                        for(let nextNeuronIndex = 0; nextNeuronIndex < this.layers[layerIndex + 1].neurons.length; nextNeuronIndex++){
                            error += gradients[layerIndex][nextNeuronIndex] * this.layers[layerIndex + 1].neurons[nextNeuronIndex].weights[neuronIndex];
                            // layerGradients[layerIndex] et pas layerGradients[layerIndex + 1] car le tableau des gradients ne contient pas la couche d'entrée
                        }
                    }
                    // Calcul de la dérivée de la fonction d'activation
                    let dsigmoidValue = this.layers[layerIndex].activationFunction.der(neuralNetworkOutputs[layerIndex][neuronIndex]);
                    // Calcul du gradient
                    layerGradients.push(error * dsigmoidValue);
                }
                gradients[layerIndex - 1] = layerGradients;
            }

            // Calcul de la modification des poids et des bias
            // Pour chaque couche (sans l'entrée)
            for(let layerIndex = 1; layerIndex < this.layers.length; layerIndex++){
                let layerIndexWithoutInputLayer = layerIndex - 1;
                // Pour chaque neurone de la couche
                for(let neuronIndex = 0; neuronIndex < this.layers[layerIndex].neurons.length; neuronIndex++){
                    // Poids
                    // Pour chaque poids du neurone
                    for(let weightIndex = 0; weightIndex < this.layers[layerIndex].neurons[neuronIndex].weights.length; weightIndex++){
                        // Learning rate * gradient du neurone * Sortie de la couche précédente
                        weightsModification[layerIndexWithoutInputLayer][neuronIndex][weightIndex] += this.learningRate * gradients[layerIndexWithoutInputLayer][neuronIndex] * neuralNetworkOutputs[layerIndex - 1][weightIndex];
                    }
                    // Bias
                    biasesModification[layerIndexWithoutInputLayer][neuronIndex] += this.learningRate * gradients[layerIndexWithoutInputLayer][neuronIndex] * 1; // x 1 car le bias a une valeur de 1
                }
            }
        }

        // Une fois que le batch d'entrée et le batch de sortie a été entièrement parcouru, on modifie les poids et les bias
        // Pour chaque couche (sauf celle d'entrée)
        for(let layerIndex = 1; layerIndex < this.layers.length; layerIndex++){
            let layerIndexWithoutInputLayer = layerIndex - 1;
            //Pour chaque neurone de la couche
            for(let neuronIndex = 0; neuronIndex < this.layers[layerIndex].neurons.length; neuronIndex++){
                // Poids
                // Pour chaque poids du neurone
                for(let weightIndex = 0; weightIndex < this.layers[layerIndex].neurons[neuronIndex].weights.length; weightIndex++){
                    this.layers[layerIndex].neurons[neuronIndex].weights[weightIndex] += weightsModification[layerIndexWithoutInputLayer][neuronIndex][weightIndex];
                }
                // Bias
                this.layers[layerIndex].neurons[neuronIndex].bias += biasesModification[layerIndexWithoutInputLayer][neuronIndex];
            }
        }
    }
}