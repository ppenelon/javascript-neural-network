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
        addData: document.getElementById('addData'),
        train: document.getElementById('train'),
        batchSize: document.getElementById('batchSize'),
        batchCount: document.getElementById('batchCount')
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
    // TODO : momentum + learning rate evolutif + calcul erreur

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
    elements.addData.onclick = function(){
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
    elements.train.onclick = function(){
        // On récupère les paramètres de batchs
        let batchSize = parseInt(elements.batchSize.value);
        let batchCount = parseInt(elements.batchCount.value);
        // Création de batchCount batchs de taille batchSize
        for(let i = 0; i < batchCount; i++){ // i -> Index du batch
            let batchInput = [];
            let batchOutput = [];
            for(let j = 0; j < batchSize; j++){ // j -> Index de l'élément (inputs) dans le batch
                let inputs = [];
                let outputs = [];
                // On récupère l'index d'une donnée d'entrainement au hasard
                let trainingDataIndex = Math.floor(Math.random() * (elements.tableData.children[0].childElementCount - 1)) + 1; // Le +1 évite le <tr> avec les <th>
                // Sauvegarde des entrées de cette donnée d'entrainement
                for(let inputTrainingData of elements.tableData.children[0].children[trainingDataIndex].children[0].children){
                    inputs.push(parseFloat(inputTrainingData.value));
                }
                // Sauvegarde des sorties de cette donnée d'entrainement
                for(let outputTrainingData of elements.tableData.children[0].children[trainingDataIndex].children[1].children){
                    outputs.push(parseFloat(outputTrainingData.value));
                }
                // Ajout de la données d'entrainement au batch
                batchInput.push(inputs);
                batchOutput.push(outputs);
            }
            // Envoi du batch au réseau de neurones pour l'entrainement
            neuralNetwork.learn(batchInput, batchOutput);
        }
    }
}