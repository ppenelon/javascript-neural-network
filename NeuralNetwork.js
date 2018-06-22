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

class NeuralNetwork{

    // learningRate; // Taux d'apprentissage

    // inputLayer; // Couche d'entrée
    // hiddenLayer; // Couche cachée
    // outputLayer; // Couche de sortie

    // activationFunction; // Fonction d'activation choisie

    constructor(nbInput, nbHidden, nbOutput, learningRate = 0.03, activationFunction = ActivationFunction.sigmoid()){
        // Création des neurones de la couche d'entrée
        this.inputLayer = [];
        for(let i = 0; i < nbInput; i++){
            this.inputLayer.push(new Neuron(0));
        }
        // Création des neurones de la couche cachée
        this.hiddenLayer = [];
        for(let i = 0; i < nbHidden; i++){
            this.hiddenLayer.push(new Neuron(nbInput));
        }
        // Création des neurones de la couche de sortie
        this.outputLayer = [];
        for(let i = 0; i < nbOutput; i++){
            this.outputLayer.push(new Neuron(nbHidden));
        }
        // Taux d'apprentissage
        this.learningRate = learningRate;
        // Fonction d'activation
        this.activationFunction = activationFunction;

        console.log(this);
    }

    // Evalue le resultat de valeur dans le réseau de neurones
    evaluate(inputs){
        // Calcul de la sortie des neurones de la couche cachée
        let hiddenOutput = [];
        for(let hiddenNeuron of this.hiddenLayer){
            hiddenOutput.push(hiddenNeuron.evaluate(inputs, this.activationFunction));
        }
        // Calcul de la sortie des neurones de la couche de sortie
        let outputOutput = [];
        for(let outputNeuron of this.outputLayer){
            outputOutput.push(outputNeuron.evaluate(hiddenOutput, this.activationFunction));
        }
        // On retourne la sortie des neurones de toutes les couches du réseau de neurones
        return [
            inputs,
            hiddenOutput,
            outputOutput
        ];
    }

    learn(batchsInput, batchsOutput){
        // On crée les tableaux qui vont recevoir les sommes de modification des poids et des bias
        let weightsOutputHiddenModification = [];
        let biasesOutputModification = [];
        for(let i = 0; i < this.outputLayer.length; i++){ // i -> Index du neurone de sortie
            // Poids
            let outputNeuronToHiddenNeurons = [];
            for(let j = 0; j < this.hiddenLayer.length; j++){ // j -> Index du neurone caché
                outputNeuronToHiddenNeurons.push(0);
            }
            weightsOutputHiddenModification.push(outputNeuronToHiddenNeurons);
            // Bias
            biasesOutputModification.push(0);
        }
        let weightsHiddenInputModification = [];
        let biasesHiddenModification = [];
        for(let i = 0; i < this.hiddenLayer.length; i++){ // i -> Index du neurone caché
            // Poids
            let hiddenNeuronToInputNeurons = [];
            for(let j = 0; j < this.inputLayer.length; j++){ // j -> Index du neurone d'entrée
                hiddenNeuronToInputNeurons.push(0);
            }
            weightsHiddenInputModification.push(hiddenNeuronToInputNeurons);
            // Bias
            biasesHiddenModification.push(0);
        }

        // On parcours tous les batchs
        for(let batchIndex = 0; batchIndex < batchsInput.length; batchIndex++){
            // On récupère les entrées/sorties de ce batch
            let inputs = batchsInput[batchIndex];
            let outputs = batchsOutput[batchIndex];

            // Calcul du résultat du réseau de neurones pour ces entrées
            let neuralNetworkOutputs = this.evaluate(inputs);
            let inputOutputs = neuralNetworkOutputs[0];
            let hiddenOutputs = neuralNetworkOutputs[1];
            let outputOutputs = neuralNetworkOutputs[2];
            
            // Gradients de la couche de sortie
            let gradientsOutput = [];
            for(let i = 0; i < this.outputLayer.length; i++){ // i -> Index du neurone de sortie
                // Calcul de l'erreur au niveau du neurone de sortie
                let error = outputs[i] - outputOutputs[i]; // cible - sortie (outputs[i] - outputOutputs[i])
                // Calcul de la dérivée de la fonction d'activation
                let dsigmoidValue = this.activationFunction.der(outputOutputs[i]);
                // Calcul du gradient
                gradientsOutput.push(error * dsigmoidValue);
            }

            // Gradient de la couche cachée
            let gradientsHidden = [];
            for(let i = 0; i < this.hiddenLayer.length; i++){ // i -> Index du neurone cachée
                // Calcul de l'erreur (somme de l'erreur des sorties * les poids associés entre le caché et la sortie)
                let error = 0;
                for(let j = 0; j < this.outputLayer.length; j++){ // j -> Index du neurone de sortie
                    error += gradientsOutput[j] * this.outputLayer[j].weights[i];
                }
                // Calcul de la dérivée partielle de la valeur envoyée par ce neurone caché
                let dsigmoidValue = this.activationFunction.der(hiddenOutputs[i]);
                // Calcul du gradient
                gradientsHidden.push(error * dsigmoidValue);
            }

            // Modification des poids et des bias des neurones de sortie
            for(let i = 0; i < this.outputLayer.length; i++){ // i -> Index du neurone de sortie
                // Poids
                for(let j = 0; j < this.hiddenLayer.length; j++){ // j -> Index du neurone caché
                    weightsOutputHiddenModification[i][j] += this.learningRate * gradientsOutput[i] * hiddenOutputs[j];
                }
                // Bias
                biasesOutputModification[i] += this.learningRate * gradientsOutput[i];
            }

            // Modification des poids et des bias des neurones cachés
            for(let i = 0; i < this.hiddenLayer.length; i++){ // i -> Index du neurone caché
                // Poids
                for(let j = 0; j < this.inputLayer.length; j++){ // j -> Index du neurone d'entrée
                    weightsHiddenInputModification[i][j] += this.learningRate * gradientsHidden[i] * inputOutputs[j];
                }
                // Bias
                biasesHiddenModification[i] += this.learningRate * gradientsHidden[i];
            }

            console.log(this);
        }

        // On applique les modifications calculées aux poids et aux bias
        for(let i = 0; i < this.outputLayer.length; i++){ // i -> Index du neurone de sortie
            for(let j = 0; j < this.hiddenLayer.length; j++){ // j -> Index du neurone caché
                this.outputLayer[i].weights[j] += weightsOutputHiddenModification[i][j];
            }
            this.outputLayer[i].bias += biasesOutputModification[i];
        }
        for(let i = 0; i < this.hiddenLayer.length; i++){ // i -> Index du neurone caché
            for(let j = 0; j < this.inputLayer.length; j++){ // j -> Index du neurone d'entrée
                this.hiddenLayer[i].weights[j] += weightsHiddenInputModification[i][j];
            }
            this.hiddenLayer[i].bias += biasesHiddenModification[i];
        }
    }
}