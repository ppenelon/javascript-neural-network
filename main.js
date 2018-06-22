window.onload = function(){
    // Debug
    document.getElementById('printNeuralNetwork').onclick = function(){ console.log(neuralNetwork); }

    // Elements HTML
    let elements = {
        addHiddenLayer: document.getElementById('addHiddenLayer'),
        deleteHiddenLayer: document.getElementById('deleteHiddenLayer'),
        tableNeuralNetworksLayersSettings: document.getElementById('tableNeuralNetworksLayersSettings'),
        inputLayerSize: document.getElementById('inputLayerSize'),
        hiddenLayerSize: document.getElementById('hiddenLayerSize'),
        outputLayerSize: document.getElementById('outputLayerSize'),
        learningRate: document.getElementById('learningRate'),
        activationFunction: document.getElementById('activationFunction'),
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
    let neuralNetwork = new NeuralNetwork(0); // La création du réseau ici (alors qu'il est recréé juste en dessous) permet d'avoir accès à l'auto-completion
    startNewNeuralNetwork();

    // Ajoute une couche cachée dans les paramètres du nouveau réseau de neurones
    elements.addHiddenLayer.onclick = function(){
        // Récupération du nombre de colonnes
        let columnsCount = elements.tableNeuralNetworksLayersSettings.children[0].children[0].childElementCount;
        // Ajout du th hidden
        let thHidden = document.createElement('th');
        thHidden.innerText = 'Hidden';
        elements.tableNeuralNetworksLayersSettings.children[0].children[0].insertBefore(
            thHidden,
            elements.tableNeuralNetworksLayersSettings.children[0].children[0].children[columnsCount - 1]
        );
        // Ajout de l'input
        let inputLayerSizeCell = document.createElement('td');
        let inputLayerSize = document.createElement('input');
        inputLayerSize.type = 'text';
        inputLayerSize.value = '1';
        inputLayerSize.size = 6;
        inputLayerSize.style = 'text-align: center;';
        inputLayerSizeCell.appendChild(inputLayerSize);
        elements.tableNeuralNetworksLayersSettings.children[0].children[1].insertBefore(
            inputLayerSizeCell,
            elements.tableNeuralNetworksLayersSettings.children[0].children[1].children[columnsCount - 1]
        );
        // Ajout de la sélection de la fonction d'activation
        let activationFunctionCell = document.createElement('td');
        let activationFunctionSelect = document.createElement('select');
        let activationFunctionSigmoidOption = document.createElement('option');
        activationFunctionSigmoidOption.value = 'sigmoid';
        activationFunctionSigmoidOption.innerText = 'Sigmoid';
        activationFunctionSelect.appendChild(activationFunctionSigmoidOption);
        let activationFunctionReluOption = document.createElement('option');
        activationFunctionReluOption.value = 'relu';
        activationFunctionReluOption.innerText = 'ReLU';
        activationFunctionSelect.appendChild(activationFunctionReluOption);
        let activationFunctionLinearOption = document.createElement('option');
        activationFunctionLinearOption.value = 'linear';
        activationFunctionLinearOption.innerText = 'Linear';
        activationFunctionSelect.appendChild(activationFunctionLinearOption);
        activationFunctionSelect.value = 'sigmoid';
        activationFunctionCell.appendChild(activationFunctionSelect);
        elements.tableNeuralNetworksLayersSettings.children[0].children[2].insertBefore(
            activationFunctionCell,
            elements.tableNeuralNetworksLayersSettings.children[0].children[2].children[columnsCount - 1]
        );
    }

    // Suppression de la dernière couche cachée dans les paramètres du nouveau réseau de neurones
    elements.deleteHiddenLayer.onclick = function(){
        // Récupération du nombre de colonnes
        let columnsCount = elements.tableNeuralNetworksLayersSettings.children[0].children[0].childElementCount;
        if(columnsCount < 4){ // Nom des lignes + Input + Hidden + Output
            return;
        }
        // Suppression de la colonne n - 1
        for(let i = 0; i < 3; i++){
            elements.tableNeuralNetworksLayersSettings.children[0].children[i].removeChild(
                elements.tableNeuralNetworksLayersSettings.children[0].children[i].children[columnsCount - 2]
            );
        }
    }

    // Fonction qui redemarre un nouveau réseau de neurones avec des paramètres précis
    elements.startNewNeuralNetwork.onclick = function(){
        // Création du réseau de neurones
        startNewNeuralNetwork();
    }

    // Réinitialise les données du réseau avec les nouvelles valeurs
    // TODO : momentum + learning rate evolutif + calcul erreur

    // Crée un nouvel réseau neuronal et remplace l'actuel
    function startNewNeuralNetwork(){
        // Taille de la couche d'entrée et de la couche de sortie du réseau de neurones
        let inputLayerSize;
        let outputLayerSize;
        // Création du nouveau réseau
        neuralNetwork = new NeuralNetwork(parseFloat(elements.learningRate.value));
        for(let i = 1; i < elements.tableNeuralNetworksLayersSettings.children[0].children[0].childElementCount; i++){
            // On récupère la taille de la couche (= le nombre de neurone)
            let layerSize = parseInt(elements.tableNeuralNetworksLayersSettings.children[0].children[1].children[i].children[0].value);
            if(i == 1){ // Si c'est la première couche, on sauvegarde sa taille
                inputLayerSize = layerSize;
            }
            outputLayerSize = layerSize; // On écrase à chaque fois la taille de la couche de sortie avec la nouvelle couche (à la dernière couche on aura la couche de sortie ;))
            // On récupère la fonction d'activation de la couche (si ce n'est pas la première couche)
            let activationFunction = ActivationFunction.sigmoid();
            if(i > 1){ // Si ce n'est pas la couche d'entrée on récupère la fonction d'activation
                let activationFunctionString = elements.tableNeuralNetworksLayersSettings.children[0].children[2].children[i].children[0].value;
                if(activationFunctionString === 'sigmoid'){
                    activationFunction = ActivationFunction.sigmoid();
                }
                else if(activationFunctionString === 'relu'){
                    activationFunction = ActivationFunction.relu();
                }
                else if(activationFunctionString === 'linear'){
                    activationFunction = ActivationFunction.linear();
                }
            }
            // On ajoute la nouvelle couche
            neuralNetwork.addLayer(layerSize, activationFunction)
        }
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
        // Taille des couches du réseau de neurones
        let inputLayerSize = neuralNetwork.layers[0].neurons.length;
        let outputLayerSize = neuralNetwork.layers[neuralNetwork.layers.length - 1].neurons.length;
        // Nouvelle ligne
        let newRow = elements.tableData.insertRow();
        // Cellule qui regroupe les inputs
        let inputCell = newRow.insertCell();
        let inputsInputValue = [];
        for(let i = 0; i < inputLayerSize; i++){
            inputsInputValue.push(document.createElement('input'));
            inputsInputValue[i].type = 'text';
            inputsInputValue[i].size = 5;
            inputsInputValue[i].value = 0;
            inputCell.appendChild(inputsInputValue[i]);
        }
        // Cellule qui regroupe les outputs
        let outputCell = newRow.insertCell();
        let inputsOutputValue = [];
        for(let i = 0; i < outputLayerSize; i++){
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
            for(let i = 0; i < inputLayerSize; i++){
                elementsInput[i].value = neuralNetworkOutputs[0][i];
            }
            // On affiche les sorties du réseau neuronal
            for(let i = 0; i < outputLayerSize; i++){
                elementsOutput[i].value = neuralNetworkOutputs[neuralNetworkOutputs.length - 1][i];
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