package NeuralTensor;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.Random;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLNumericArray;
import com.sun.xml.internal.bind.v2.model.core.ArrayInfo;

import edu.stanford.nlp.optimization.QNMinimizer;
import edu.umass.nlp.optimize.IDifferentiableFn;
import edu.umass.nlp.optimize.IOptimizer;
import edu.umass.nlp.optimize.LBFGSMinimizer;
import edu.umass.nlp.utils.BasicPair;
import edu.umass.nlp.utils.DoubleArrays;
import edu.umass.nlp.utils.IPair;
public class Run_NTN {

    public static void main(String[] args) throws IOException {

        INDArray x = Nd4j.rand(new int[]{2, 3,3});
        INDArray y = Nd4j.rand(3,4);
        INDArray z = Nd4j.create(3,2);
        z.putColumn(0, y.getColumn(0));
        z.putColumn(1, y.getColumn(2));
        System.out.println("x s0:" + x.slice(0));
        System.out.println(x.slice(0).mmul(z));

        INDArray sliceOfx=Nd4j.create(3,3);
        for (int i = 0; i < sliceOfx.linearView().length(); i++) {
            sliceOfx.put(i, x.slice(0).linearView().getScalar(i));
        }
        System.out.println("slice :"+sliceOfx);
        System.out.println(sliceOfx.mmul(z));

        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        Random rand = new Random();

        //Data path
        String data_path="";String theta_save_path=""; String theta_load_path="";
        try {
            data_path = args[0];
            theta_load_path = args[1];
            theta_save_path = args[2];
        } catch (Exception e) {
            data_path = "D:\@PUCP\@Tesis\appCentral\src\main\java\NeuralTensor";
            theta_save_path = "D:\@PUCP\@Tesis\appCentral\src\main\java\NeuralTensor";
        }

        int batchSize = 20000; 				
        int numWVdimensions = 100; 		
        int numIterations = 500; 			
        int batch_iterations = 5;			
        int sliceSize = 3; 					
        int corrupt_size = 10; 				
        String activation_function= "tanh"; 
        float lamda = 0.0001F;			
        boolean optimizedLoad=false;	
        boolean reducedNumOfRelations = false; 
        boolean minimizer = true;			
        boolean german = false;

        System.out.println("NTN: batches: "+batchSize+" | slice: "+sliceSize+" | numero iteraciones:"+numIterations+" | dimensiones corruptos: "+corrupt_size+"| funcion de actividad: "+ activation_function);

        
        Util u = new Util();
        //carga de archivos para entrenamiento
        DataFactory tbj = DataFactory.getInstance(batchSize, corrupt_size, numWVdimensions, german, reducedNumOfRelations);
        tbj.loadEntitiesFromSocherFile(data_path +"entities.txt");
        tbj.loadRelationsFromSocherFile(data_path + "relations.txt");
        tbj.loadTrainingDataTripplesE1rE2(data_path + "train.txt");
        tbj.loadDevDataTripplesE1rE2Label(data_path + "dev.txt");
        tbj.loadTestDataTripplesE1rE2Label(data_path + "test.txt");


        // Creacion de la red RNTN
        NTN t = new NTN(numWVdimensions, tbj.getNumOfentities(), tbj.getNumOfRelations(), tbj.getNumOfWords(), batchSize, sliceSize, activation_function, tbj, lamda);
        t.connectDatafactory(tbj);


        double[] theta = t.getTheta_inital().data().asDouble();

        //Entrenamiento

        for (int i = 0; i < numIterations; i++) {

            tbj.generateNewTrainingBatchJob();

            if (minimizer == true) {
                LBFGSMinimizer.Opts optimizerOpts = new LBFGSMinimizer.Opts();
                optimizerOpts.maxIters=batch_iterations;
                IOptimizer.Result res = (new LBFGSMinimizer()).minimize(t, theta, optimizerOpts);
                System.out.println("res: "+res.didConverge + "| "+res.minObjVal);
                theta = res.minArg;
            }else{
                QNMinimizer qn = new QNMinimizer() ;
                qn.terminateOnMaxItr(batch_iterations);
                theta = qn.minimize(t, 1e-4, theta);
            }
            System.out.println("Paramters for batchjob optimized, iteration: "+i+" completed");

            if (i==5 || i%10==0) {
                Nd4j.writeTxt( u.convertDoubleArrayToFlattenedINDArray(theta), theta_save_path+"//theta_opt_iteration_"+i+".txt", ",");
            }

        }

        Nd4j.writeTxt(u.convertDoubleArrayToFlattenedINDArray(theta) , theta_save_path+"//theta_opt"+Calendar.getInstance().get(Calendar.DATE)+".txt", ",");
        System.out.println("Model saved!");

        //Test
        System.out.println("Precision "+theta_load_path);

        INDArray best_theresholds = t.computeBestThresholds(u.convertDoubleArrayToFlattenedINDArray(theta), tbj.getDevTripples());
        System.out.println("Mejores : "+best_theresholds);

        INDArray predictions = t.getPrediction(u.convertDoubleArrayToFlattenedINDArray(theta), tbj.getTestTripples(), best_theresholds);
    }
}
