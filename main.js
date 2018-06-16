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
        this.bias = (Math.random()) - 1;
    }

    // Calcule la sortie du neurone
    evaluate(inputs){
        // Somme des valeurs d'entrées * les poids
        let value = 0;
        for(let i = 0; i < this.weights.length; i++){
            value += inputs[i] * this.weights[i];
        }
        // On ajoute le bias
        value += this.bias;
        // On applique la fonction d'activation
        return NeuralNetwork.sigmoid(value);
    }
}

class NeuralNetwork{

    // learningRate; // Taux d'apprentissage

    // inputLayer; // Couche d'entrée
    // hiddenLayer; // Couche cachée
    // outputLayer; // Couche de sortie

    constructor(nbInput, nbHidden, nbOutput, learningRate){
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
    }

    // Evalue le resultat de valeur dans le réseau de neurones
    evaluate(inputs){
        // Calcul de la sortie des neurones de la couche cachée
        let hiddenOutput = [];
        for(let hiddenNeuron of this.hiddenLayer){
            hiddenOutput.push(hiddenNeuron.evaluate(inputs));
        }
        // Calcul de la sortie des neurones de la couche de sortie
        let outputOutput = [];
        for(let outputNeuron of this.outputLayer){
            outputOutput.push(outputNeuron.evaluate(hiddenOutput));
        }
        // On retourne la sortie des neurones de toutes les couches du réseau de neurones
        return [
            inputs,
            hiddenOutput,
            outputOutput
        ];
    }

    learn(inputs, outputs){
        // Calcul du résultat du réseau de neurones pour ces entrées
        let neuralNetworkOutputs = this.evaluate(inputs);
        let inputOutputs = neuralNetworkOutputs[0];
        let hiddenOutputs = neuralNetworkOutputs[1];
        let outputOutputs = neuralNetworkOutputs[2];

        // Gradients de la couche de sortie
        let gradientsOutput = [];
        for(let i = 0; i < this.outputLayer.length; i++){ // i -> Index du neurone de sortie
            // Calcul de l'erreur au niveau du neurone de sortie
            let error = outputs[i] - outputOutputs[i];
            // Calcul de la dérivée de la fonction d'activation
            let dsigmoidValue = NeuralNetwork.dsigmoid(outputOutputs[i]);
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
            let dsigmoidValue = NeuralNetwork.dsigmoid(hiddenOutputs[i]);
            // Calcul du gradient
            gradientsHidden.push(error * dsigmoidValue);
        }

        // Modification des poids et des bias des neurones de sortie
        for(let i = 0; i < this.outputLayer.length; i++){ // i -> Index du neurone de sortie
            // Poids
            for(let j = 0; j < this.hiddenLayer.length; j++){ // j -> Index du neurone caché
                this.outputLayer[i].weights[j] += this.learningRate * gradientsOutput[i] * hiddenOutputs[j];
            }
            // Bias
            this.outputLayer[i].bias += this.learningRate * gradientsOutput[i];
        }

        // Modification des poids et des bias des neurones cachés
        for(let i = 0; i < this.hiddenLayer.length; i++){ // i -> Index du neurone caché
            // Poids
            for(let j = 0; j < this.inputLayer.length; j++){ // j -> Index du neurone d'entrée
                this.hiddenLayer[i].weights[j] += this.learningRate * gradientsHidden[i] * inputOutputs[j];
            }
            // Bias
            this.hiddenLayer[i].bias += this.learningRate * gradientsHidden[i];
        }
    }

    // Fonction d'activation
    static sigmoid(x){
        return 1 / (1 + Math.exp(-1 * x));
    }

    // Dérivée de la fonction d'activation (sachant que le paramètre y est une valeur déjà passée dans la fonction d'activation)
    static dsigmoid(y){
        return y * (1 - y);
    }
}

window.onload = function(){
    // Elements HTML
    let elements = {
        inputLayerSize: document.getElementById('inputLayerSize'),
        hiddenLayerSize: document.getElementById('hiddenLayerSize'),
        outputLayerSize: document.getElementById('outputLayerSize'),
        learningRate: document.getElementById('learningRate'),
        startNewNeuralNetwork: document.getElementById('startNewNeuralNetwork'),
        inputNeuronsValue: document.getElementById('inputNeuronsValue'),
        outputNeuronsValue: document.getElementById('outputNeuronsValue'),
        tableData: document.getElementById('tableData'),
        more: document.getElementById('more'),
        loop: document.getElementById('loop'),
        loopCount: document.getElementById('loopCount')
    }

    // Elements HTML des neurones
    let elementsInput = [];
    let elementsOutput = [];

    // Réseau de neurones
    let neuralNetwork = new NeuralNetwork(2, 2, 1, 0.01); // La création du réseau ici (alors qu'il est recréé juste en dessous) permet d'avoir accès à l'auto-completion
    startNewNeuralNetwork(2, 2, 1, 0.1);

    // Affichage des données du réseau de neurones
    elements.inputLayerSize.value = neuralNetwork.inputLayer.length;
    elements.hiddenLayerSize.value = neuralNetwork.hiddenLayer.length;
    elements.outputLayerSize.value = neuralNetwork.outputLayer.length;
    elements.learningRate.value = neuralNetwork.learningRate;

    // Fonction qui redemarre un nouveau réseau de neurones avec des paramètres précis
    elements.startNewNeuralNetwork.onclick = function(){
        startNewNeuralNetwork(
            parseInt(elements.inputLayerSize.value),
            parseInt(elements.hiddenLayerSize.value),
            parseInt(elements.outputLayerSize.value),
            parseFloat(elements.learningRate.value)
        );
    }

    // Réinitialise les données du réseau avec les nouvelles valeurs
    // TODO : momentum + batch size + learning rate evolutif

    // Crée un nouvel réseau neuronal et remplace l'actuel
    function startNewNeuralNetwork(inputLayerSize, hiddenLayerSize, outputLayerSize, learningRate){
        // Création du nouveau réseau
        neuralNetwork = new NeuralNetwork(inputLayerSize, hiddenLayerSize, outputLayerSize, learningRate);
        // Modification du réseau de neurones visuels
        elementsInput = []; // Références vers les éléments DOM
        while(elements.inputNeuronsValue.firstChild) elements.inputNeuronsValue.removeChild(elements.inputNeuronsValue.firstChild); // On supprime tous les inputs de la valeur des neurones d'entrée
        for(let i = 0; i < inputLayerSize; i++){ // On ajoute autant d'input que de neuronne d'entrée
            let newInputNeuronValueElement = document.createElement('input');
            newInputNeuronValueElement.type = 'text';
            newInputNeuronValueElement.readOnly = true;
            newInputNeuronValueElement.className = 'neuron-input';
            elementsInput.push(newInputNeuronValueElement);
            elements.inputNeuronsValue.appendChild(newInputNeuronValueElement);
        }
        elementsOutput = []; // Idem
        while(elements.outputNeuronsValue.firstChild) elements.outputNeuronsValue.removeChild(elements.outputNeuronsValue.firstChild); // Idem
        for(let i = 0; i < outputLayerSize; i++){ // Idem
            let newOutputNeuronValueElement = document.createElement('input');
            newOutputNeuronValueElement.type = 'text';
            newOutputNeuronValueElement.readOnly = true;
            newOutputNeuronValueElement.className = 'neuron-input';
            elementsOutput.push(newOutputNeuronValueElement);
            elements.outputNeuronsValue.appendChild(newOutputNeuronValueElement);
        }
        // Remise à zéro du tableau de données d'entrées
        let rowsCount = elements.tableData.children[0].childElementCount;
        for(let i = 1; i < rowsCount; i++){
            elements.tableData.deleteRow(1);
        }
    }

    // Ajoute des lignes au tableau
    elements.more.onclick = function(){
        // Nouvelle ligne
        let newRow = elements.tableData.insertRow();
        // Cellule qui regroupe les inputs
        let inputCell = newRow.insertCell();
        let inputsInputValue = [];
        for(let i = 0; i < neuralNetwork.inputLayer.length; i++){
            inputsInputValue.push(document.createElement('input'));
            inputsInputValue[i].type = 'text';
            inputsInputValue[i].size = 5;
            inputsInputValue[i].value = 0;
            inputCell.appendChild(inputsInputValue[i]);
        }
        // Cellule qui regroupe les outputs
        let outputCell = newRow.insertCell();
        let inputsOutputValue = [];
        for(let i = 0; i < neuralNetwork.outputLayer.length; i++){
            inputsOutputValue.push(document.createElement('input'));
            inputsOutputValue[i].type = 'text';
            inputsOutputValue[i].size = 5;
            inputsOutputValue[i].value = 0;
            outputCell.appendChild(inputsOutputValue[i]);
        }
        // Cellule qui contient le bouton d'entrainement
        let trainCell = newRow.insertCell();
        let trainButton = document.createElement('button');
        trainButton.innerText = 'Train';
        trainButton.onclick = function(){
            // On récupère les entrées
            let inputs = [];
            for(let inputElement of inputsInputValue){
                inputs.push(inputElement.value);
            }
            // On récupère les sorties
            let outputs = [];
            for(let outputElement of inputsOutputValue){
                outputs.push(outputElement.value);
            }
            // On entraine le réseau
            neuralNetwork.learn(inputs, outputs);
        }
        trainCell.appendChild(trainButton);
        // Cellule qui contient le bouton de determination
        let determineCell = newRow.insertCell();
        let determineButton = document.createElement('button');
        determineButton.innerText = 'Determine';
        determineButton.onclick = function(){
            // On récupère les entrées
            let inputs = [];
            for(let inputElement of inputsInputValue){
                inputs.push(inputElement.value);
            }
            // On récupère le résultat du réseau neuronal sur les valeurs d'entrée
            let neuralNetworkOutputs = neuralNetwork.evaluate(inputs);
            // On affiche les entrées du réseau neuronal
            for(let i = 0; i < neuralNetwork.inputLayer.length; i++){
                elementsInput[i].value = neuralNetworkOutputs[0][i];
            }
            // On affiche les sorties du réseau neuronal
            for(let i = 0; i < neuralNetwork.outputLayer.length; i++){
                elementsOutput[i].value = neuralNetworkOutputs[2][i];
            }
        }
        determineCell.appendChild(determineButton);
        // Cellule qui contient le bouton de suppression
        let deleteCell = newRow.insertCell();
        let deleteButton = document.createElement('button');
        deleteButton.innerText = 'Delete';
        deleteButton.onclick = function(){
            for(let i = 0; i < elements.tableData.children[0].childElementCount; i++){
                if(elements.tableData.children[0].children[i] == newRow){
                    elements.tableData.deleteRow(i);
                    break;
                }
            }
        }
        deleteCell.appendChild(deleteButton);
    }

    // Boucle sur les données d'entrainement
    elements.loop.onclick = function(){
        // On regarde si le nombre de boucle est bien un nombre et supérieur à 0
        let loopCount = parseInt(elements.loopCount.value);
        if(isNaN(loopCount) || loopCount <= 0){
            return;
        }
        // On lance l'entrainement loopCount fois
        for(let i = 0; i < loopCount; i++){
            let trainingDataIndex = Math.floor(Math.random() * (elements.tableData.children[0].childElementCount - 1)) + 1; // Le +1 évite le <tr> avec les <th>
            elements.tableData.children[0].children[trainingDataIndex].children[2].children[0].onclick(); // On simule le click sur cette fonction d'entrainement
        }
    }
}