/**
 * 
 */
package deeplearning4j;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * @author Amit Kumar
 *
 */
public class RoomOccupancyDetector {
	public static void main(String[] args) throws Exception {
		final int seed = 192; // seed for generation of random numbers
		final int batchSize = 10;
		
		final int numInputs = 7;
		final int numHiddenNodes = 50;
		final int numOutputs = 2;
		
		final int nEpochs = 10;

		// Load training data
		final RecordReader trainingRecordReader = new CSVRecordReader(1); // Skip header line to avoid number format exception
		trainingRecordReader.initialize(new FileSplit(new File("src/main/resources/occupancy_data/datatraining.txt")));
		final DataSetIterator trainIterator = new RecordReaderDataSetIterator(trainingRecordReader, null, batchSize, 7, numOutputs);
		
		final MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
				.seed(seed) // random number generator seed
				.iterations(1) // Iterations for optimizations
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // To be read
				.learningRate(0.1)
				.updater(Updater.NESTEROVS) // To be read
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(numInputs) // To be read
						.nOut(numHiddenNodes)
						.weightInit(WeightInit.XAVIER) // To be read
						.activation(Activation.RELU)
						.build())
				.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // To be read
						.nIn(numHiddenNodes)
						.nOut(numOutputs)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.SOFTMAX)
						.build())
				.pretrain(false)
				.backprop(true)
				.build();

		final MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        
		for ( int n = 0; n < nEpochs; n++) {
            model.fit( trainIterator);
        }

		System.out.println("Evaluate model....");
		
		// Load test data
		final RecordReader testRecordReader = new CSVRecordReader(1); // Skip header line to avoid number format exception
		testRecordReader.initialize(new FileSplit(new File("src/main/resources/occupancy_data/dataset.txt")));
		final DataSetIterator testIterator = new RecordReaderDataSetIterator(testRecordReader, batchSize, 0, 2);
				
				
        Evaluation eval = new Evaluation(numOutputs);
        while(testIterator.hasNext()){
            DataSet t = testIterator.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);

            eval.eval(lables, predicted);

        }

        //Print the evaluation statistics
        System.out.println(eval.stats());
		
		
	}
}
