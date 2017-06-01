
import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;


/**
 *
 * @author  Jaime Diego Bustamante Arce
 * @date    15/04/2017
 */
public class ProyectoGeneral {

    public static void main(String[] args) throws FileNotFoundException, IOException {
        System.out.println("Working Directory = " +
            System.getProperty("user.dir"));


        //Extracción y estructuración de eventos

        //a. Recolección de información y b. Estructuración de eventos
        //Desde el archivo CSV obtenido mediante y luego la estructuración
        MainEstructuracion estructuracionEventos=new MainEstructuracion();
        ArrayList<String> estructuras=new ArrayList<String>();

        CSVReader reader = new CSVReader(new FileReader(".\\appCentral\\src\\main\\resources\\data_reuters.csv"), ',', '|');
        String [] nextLine;

        CSVWriter writer = new CSVWriter(new FileWriter(".\\appCentral\\src\\main\\resources\\estructures_events.csv",true), ',',CSVWriter.NO_QUOTE_CHARACTER);
        // feed in your array (or convert your data to an array)

        //Writer de los vectores
        CSVWriter writerVectWord = new CSVWriter(new FileWriter(".\\appCentral\\src\\main\\resources\\vectores_palabras.csv",true), ',','|');
        // feed in your array (or convert your data to an array)


        //a. Vectorización de Palabras mediante word2Vec


        File gModel = new File(".\\appCentral\\src\\main\\resources\\GoogleNews-vectors-negative300.bin.gz");
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(gModel);
        WeightLookupTable weightLookupTable = word2Vec.lookupTable();

        int count=0;
        while ((nextLine = reader.readNext()) != null) {
            // nextLine[] is an array of values from the line
            System.out.println(++count+".");
            System.out.println(nextLine[0] + nextLine[1]);
            /*
            if (count==101) {
                writer.close();
                writerVectWord.close();
                return;
            }*/
            estructuras.clear();
            estructuracionEventos.extraerEstructura(estructuras,nextLine[1]);
            for (int i=0;i<estructuras.size();i++){
                String dateString=nextLine[0].concat(";");
                String aux=dateString.concat(estructuras.get(i));
                String[] entries = aux.split(";");
                writer.writeNext(entries);
                //Write vectors
                String vectorWord=";";
                for(int o=1;o<4;o++){
                    int j=0;
                    String wordsInEntry[]=entries[o].split(" ");
                    INDArray wordVectorMatrixTotal= Nd4j.zeros(1,300);
                    for (j=0;j<wordsInEntry.length;j++){
                        INDArray wordVectorMatrix = word2Vec.getWordVectorMatrix(wordsInEntry[j]);
                        wordVectorMatrixTotal=wordVectorMatrixTotal.add(wordVectorMatrix);
                    }

                    wordVectorMatrixTotal=wordVectorMatrixTotal.div(j);
                    vectorWord=vectorWord.concat(wordVectorMatrixTotal.toString());
                    vectorWord=vectorWord.concat(";");
                }
                String[] entradaDeVectores=(aux.concat(vectorWord)).split(";");
                writerVectWord.writeNext(entradaDeVectores);
            }
        }
		writerData=writer.clone();
        writer.close();
        writerVectWord.close();
		
		
		//red Recurrente
		//365 layers de entrada
		int lstmLayerSize = 365;					
		int miniBatchSize = 32;						
		int exampleLength = 1000;					
        int tbpttLength = 50;                       
		int numEpochs = 1;							
        int generateSamplesEveryNMinibatches = 10;  
		int nSamplesToGenerate = 4;					
		int nCharactersToSample = 300;				
		String generationInitialization = null;		
		
		Random rng = new Random(12345);

		CharacterIterator iter = getStockIterator(miniBatchSize,exampleLength);
		int nOut = iter.totalOutcomes();

		//Configuracion de la red
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
			.learningRate(0.1)
			.rmsDecay(0.95)
			.seed(12345)
			.regularization(true)
			.l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.RMSPROP)
			.list()
			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
					.activation(Activation.TANH).build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.activation(Activation.TANH).build())
			.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.LOGISTIC)        //Clasificador logistico
					.nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
			.pretrain(false).backprop(true)
			.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		//Entrenamiento y test
        int miniBatchNumber = 0;
		for( int i=0; i<numEpochs; i++ ){
            while(iter.hasNext()){
                DataSet ds = iter.next();
                net.fit(ds);
                if(++miniBatchNumber % generateSamplesEveryNMinibatches == 0){
                    System.out.println("--------------------");
                    String[] samples = writerData.getData(i);
                    for( int j=0; j<samples.length; j++ ){
                        System.out.println("----- Sample " + j + " -----");
                        System.out.println(samples[j]);
                        System.out.println();
                    }
                }
            }

			iter.reset();	
		}
		System.out.println("Precision: "+ net.acc());
		System.out.println("MCC: "+ net.mcc());
    }

}
