package NeuralTensor;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLDouble;

public class DataFactory{
    private static DataFactory instance;

    ArrayList<Tripple> trainingTripples = new ArrayList<Tripple>(); 
    ArrayList<Tripple> devTripples = new ArrayList<Tripple>(); 
    ArrayList<Tripple> testTripples = new ArrayList<Tripple>();

    private HashMap<Integer, String> entitiesNumWord = new HashMap<Integer, String>();
    private HashMap<String, Integer> entitiesWordNum = new HashMap<String, Integer>();

    private HashMap<Integer, String> entitiesNumtranslateWord = new HashMap<Integer, String>();
    private HashMap<String, String> entitiesWordtranslateWord = new HashMap<String, String>();

    private HashMap<Integer, String> relationsNumWord = new HashMap<Integer, String>();
    private HashMap<String, Integer> relationsWordNum = new HashMap<String, Integer>();

    private HashMap<Integer, Tripple> trainingDataNumTripple = new HashMap<Integer, Tripple>();
    private HashMap<Integer, Tripple> devDataNumTripple = new HashMap<Integer, Tripple>();
    private HashMap<Integer, Tripple> testDataNumTripple = new HashMap<Integer, Tripple>();

    private HashMap<Integer, String> vocabNumWord = new HashMap<Integer, String>(); 
    private HashMap<String, Integer> vocabWordNum = new HashMap<String, Integer>();
    private static HashMap<String, INDArray> worvectorWordVec = new HashMap<String, INDArray>(); 
    private static HashMap<Integer, INDArray> worvectorNumVec = new HashMap<Integer, INDArray>();
    private INDArray wordVectorMaxtrixLoaded; 

    private ArrayList<String> vocab = new ArrayList<String>(); 
    private ArrayList<String> vocabDE = new ArrayList<String>(); 

    private HashMap<Integer, String> vocabNumWordDE = new HashMap<Integer, String>();
    private HashMap<String, Integer> vocabWordNumDE = new HashMap<String, Integer>();
    private static HashMap<String, INDArray> worvectorWordVecDE = new HashMap<String, INDArray>();
    private static HashMap<Integer, INDArray> worvectorNumVecDE = new HashMap<Integer, INDArray>();


    private int numOfentities;
    private int numOfRelations;
    private int numOfWords;
    private int batch_size;
    private int corrupt_size;
    private int embeddings_size;
    private boolean reduced_RelationSize;
    private int reduceRelationToSize=2;
    private boolean german;


    private ArrayList<Tripple> batchjob = new ArrayList<Tripple>();  // contains the data of a batch training job to optimize paramters


    public static DataFactory getInstance (int _batch_size, int _corrupt_size, int _embedding_size, boolean _reduce_RelationSize, boolean _german) {
        if (DataFactory.instance == null) {
            DataFactory.instance = new DataFactory (_batch_size, _corrupt_size, _embedding_size, _reduce_RelationSize, _german);
        }
        return DataFactory.instance;

    }

    private DataFactory(int _batch_size, int _corrupt_size,int _embedding_size, boolean _reduce_RelationSize, boolean _german){
        batch_size = _batch_size;
        corrupt_size = _corrupt_size;
        embeddings_size = _embedding_size;
        reduced_RelationSize = _reduce_RelationSize;
        german = _german;
    }
    public int getNumOfentities() {
        return numOfentities;
    }

    public ArrayList<Tripple> getBatchJobTripplesOfRelation(int _relation_index){
        ArrayList<Tripple> tripplesOfThisRelationFromBatchJob = new ArrayList<Tripple>();
        
        for (int i = 0; i < batchjob.size(); i++) {
            if (batchjob.get(i).getIndex_relation()==_relation_index) {
                tripplesOfThisRelationFromBatchJob.add(batchjob.get(i));
            }
        }
        return tripplesOfThisRelationFromBatchJob;
    }

    public ArrayList<Tripple> getTripplesOfRelation(int _relation_index, ArrayList<Tripple> _listWithTripples){
        ArrayList<Tripple> tripples = new ArrayList<Tripple>();
        
        for (int i = 0; i < _listWithTripples.size(); i++) {
            if (_listWithTripples.get(i).getIndex_relation()==_relation_index) {
                tripples.add(_listWithTripples.get(i));
            }
        }
        return tripples;
    }

    public ArrayList<Tripple> getAllTrainingTripples() {
        return trainingTripples;
    }


    public void generateNewTrainingBatchJob(){
        batchjob.clear();
        
        Random rand = new Random();
        
        for (int h = 0; h < corrupt_size; h++) {
            for (int i = 0; i < batch_size; i++) {
                int random_corrupt_entity = rand.nextInt(((numOfentities-1) - 0) + 1) + 0;
                batchjob.add(new Tripple(trainingTripples.get(i), random_corrupt_entity));
        
            }
        }

    }
    public void getEntity1vectormatrixOfBatchJob(){
    }

    public INDArray getINDArrayOfTripples(ArrayList<Tripple> _tripples){
        INDArray tripplesMatrix = Nd4j.zeros(_tripples.size(),3);

        for (int i = 0; i < _tripples.size(); i++) {
            tripplesMatrix.put(i,0, _tripples.get(i).getIndex_entity1());
            tripplesMatrix.put(i,1, _tripples.get(i).getIndex_relation());
            tripplesMatrix.put(i,2, _tripples.get(i).getIndex_entity2());
            System.out.println("tripplesMatrix: "+tripplesMatrix);
        }

        return tripplesMatrix;
    }

    public void loadEntitiesFromSocherFile(String path) throws IOException{
        FileReader fr = new FileReader(path);
        BufferedReader br = new BufferedReader(fr);
        String line = br.readLine();
        int entities_counter = 0;
        while (line != null) {
            entitiesNumWord.put(entities_counter, line);
            entitiesWordNum.put(line,entities_counter);
           
            String entity_name_clear; 
            try {
                entity_name_clear = line.substring(2, line.lastIndexOf("_"));
            } catch (Exception e) {
                entity_name_clear =line.substring(2);
            }
            

            if (entity_name_clear.contains("_")) { 
                
                for (int j = 0; j <entity_name_clear.split("_").length; j++) {
                    vocab.add(entity_name_clear.split("_")[j]);
                }
            }else{
                
                vocab.add(entity_name_clear);
            }

            line = br.readLine();
            entities_counter++;
        }
        br.close();
        
        numOfentities = entities_counter;
       
    }

    public void loadRelationsFromSocherFile(String path) throws IOException{
        
        FileReader fr = new FileReader(path);
        BufferedReader br = new BufferedReader(fr);
        String line = br.readLine();
        int relations_counter = 0;


        if (reduced_RelationSize==false) {
            while (line!=null) {
                relationsNumWord.put(relations_counter,line);
                relationsWordNum.put(line,relations_counter);
                line = br.readLine();
                relations_counter++;
            }
            numOfRelations = relations_counter;
        }else{
            while (relations_counter<reduceRelationToSize) {
                relationsNumWord.put(relations_counter,line);
                relationsWordNum.put(line,relations_counter);
                line = br.readLine();
                relations_counter++;
            }
            numOfRelations = relations_counter;
        }
        br.close();
       

    }

    public void loadTrainingDataTripplesE1rE2(String path) throws IOException{
        FileReader fr = new FileReader(path);
        BufferedReader br = new BufferedReader(fr);
        String line = br.readLine();
        int trainings_tripple_counter = 0;
        while (line != null) {
            
            if (reduced_RelationSize==false) {
                int e1 = entitiesWordNum.get(line.split("\\s")[0]);
                int rel = relationsWordNum.get(line.split("\\s")[1]);
                int e2 = entitiesWordNum.get(line.split("\\s")[2]);
                trainingDataNumTripple.put(trainings_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2] ));
                trainingTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2] ));
                line = br.readLine();
                trainings_tripple_counter++;
            }else{
                if (line.split("\\s")[1].equals(relationsNumWord.get(0))|line.split("\\s")[1].equals(relationsNumWord.get(1))) {
                    int e1 = entitiesWordNum.get(line.split("\\s")[0]);
                    int rel = relationsWordNum.get(line.split("\\s")[1]);
                    int e2 = entitiesWordNum.get(line.split("\\s")[2]);
                    trainingDataNumTripple.put(trainings_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2] ));
                    trainingTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2] ));
                    line = br.readLine();
                    trainings_tripple_counter++;
                }else{
                    line = br.readLine();
                }

            }
        }
        br.close();
        
    }

    public void loadDevDataTripplesE1rE2Label(String path) throws IOException{
        
        FileReader fr = new FileReader(path);
        BufferedReader br = new BufferedReader(fr);
        String line = br.readLine();
        int dev_tripple_counter = 0;
        while (line != null) {
            
            if (reduced_RelationSize==false) {
                int e1 = entitiesWordNum.get(line.split("\\s")[0]);
                int rel = relationsWordNum.get(line.split("\\s")[1]);
                int e2 = entitiesWordNum.get(line.split("\\s")[2]);
                int label = Integer.parseInt(line.split("\\s")[3]);
                devDataNumTripple.put(dev_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
                devTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
                line = br.readLine();
                dev_tripple_counter++;
            }else{
                if (line.split("\\s")[1].equals(relationsNumWord.get(0))) {
                    int e1 = entitiesWordNum.get(line.split("\\s")[0]);
                    int rel = relationsWordNum.get(line.split("\\s")[1]);
                    int e2 = entitiesWordNum.get(line.split("\\s")[2]);
                    int label = Integer.parseInt(line.split("\\s")[3]);
                    devDataNumTripple.put(dev_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
                    devTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
                    line = br.readLine();
                    dev_tripple_counter++;
                }else{
                    line = br.readLine();
                }
            }
        }
        br.close();
       
    }

    public void loadTestDataTripplesE1rE2Label(String path) throws IOException{
        
        FileReader fr = new FileReader(path);
        BufferedReader br = new BufferedReader(fr);
        String line = br.readLine();
        int test_tripple_counter = 0;
        while (line != null) {
            if (reduced_RelationSize==false) {
                
                int e1 = entitiesWordNum.get(line.split("\\s")[0]);
                int rel = relationsWordNum.get(line.split("\\s")[1]);
                int e2 = entitiesWordNum.get(line.split("\\s")[2]);
                int label = Integer.parseInt(line.split("\\s")[3]);
                testDataNumTripple.put(test_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
                testTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
                line = br.readLine();
                test_tripple_counter++;
            }else{
                if (line.split("\\s")[1].equals(relationsNumWord.get(0))) {
                    int e1 = entitiesWordNum.get(line.split("\\s")[0]);
                    int rel = relationsWordNum.get(line.split("\\s")[1]);
                    int e2 = entitiesWordNum.get(line.split("\\s")[2]);
                    int label = Integer.parseInt(line.split("\\s")[3]);
                    testDataNumTripple.put(test_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
                    testTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
                    line = br.readLine();
                    test_tripple_counter++;
                }else{
                    line = br.readLine();
                }
            }

        }
        br.close();
        

    }

    public void loadWordVectorsFromMatFile(String path, boolean optimizedLoad){
        
        try {
            MatFileReader matfilereader = new MatFileReader(path);
            
            MLCell words_mat = (MLCell) matfilereader.getMLArray("words");
            
            MLArray wordvectors_mat = (MLArray) matfilereader.getMLArray("We");
            MLDouble mlArrayDouble = (MLDouble) wordvectors_mat;
            System.out.println("mlArrayDouble"+mlArrayDouble.getM()+"|"+mlArrayDouble.getN()+"|"+mlArrayDouble.getNDimensions()+"|"+mlArrayDouble.getSize());
            
            String word;
            int wvCounter=0;
            for (int i = 0; i < mlArrayDouble.getSize()/100; i++) {
            
            
                word = words_mat.get(i).contentToString().substring(7,words_mat.get(i).contentToString().lastIndexOf("'"));
                vocab.add("unknown");
                if (optimizedLoad==true) {
            
                    if (vocab.contains(word)) { //look up if there is an entity with this word
                        vocabNumWord.put(wvCounter, word);
                        vocabWordNum.put(word, wvCounter);
                        INDArray wordvector = Nd4j.zeros(100,1);
                        for (int j = 0; j < 100; j++) {
                            wordvector.put(j, 0, mlArrayDouble.get(i, j));
                        }

                        worvectorNumVec.put(wvCounter, wordvector);
                        worvectorWordVec.put(word, wordvector);
                        wvCounter++;
                    }

                }else{

                    vocabNumWord.put(wvCounter, word);
                    vocabWordNum.put(word, wvCounter);
                    INDArray wordvector = Nd4j.ones(100,1);
                    for (int j = 0; j < 100; j++) {
                        wordvector.put(j, 0, mlArrayDouble.get(i, j));
                    }
                    worvectorNumVec.put(wvCounter, wordvector);
                    worvectorWordVec.put(word, wordvector);
                  
                    wvCounter++;
                }

            }
            numOfWords = worvectorNumVec.size();
           
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
      
        wordVectorMaxtrixLoaded = Nd4j.zeros(100, numOfWords);
        for (int i = 0; i < numOfWords; i++) {
      
            wordVectorMaxtrixLoaded.putColumn(i, worvectorNumVec.get(i));
        }
      
    }

    public INDArray createVectorsForEachEntityByWordVectors(){
      
        INDArray entity_vectors = Nd4j.zeros(embeddings_size,numOfentities);
        for (int i = 0; i < entitiesNumWord.size(); i++) {
            String entity_name; 
            try {
                entity_name = entitiesNumWord.get(i).substring(2, entitiesNumWord.get(i).lastIndexOf("_"));
            } catch (Exception e) {
                entity_name =entitiesNumWord.get(i).substring(2);
            }
          

            if (entity_name.contains("_")) { 
                
                INDArray entityvector = Nd4j.zeros(embeddings_size, 1);
                int counterOfWordVecs = 0;

                for (int j = 0; j < entity_name.split("_").length; j++) {
                    try {
                        entityvector = entityvector.add(worvectorWordVec.get(entity_name.split("_")[j]));
                    } catch (Exception e) {
                        
                        entityvector = entityvector.add(worvectorWordVec.get("unknown"));
                    }
                    counterOfWordVecs++;
                }

                entityvector = entityvector.div(counterOfWordVecs);
                entity_vectors.putColumn(i, entityvector);
            }else{
              
                try {
                    entity_vectors.putColumn(i, worvectorWordVec.get(entity_name));
                } catch (Exception e) {
                 
                    entity_vectors.putColumn(i, worvectorWordVec.get("unknown"));
                }
            }
        }
        return entity_vectors;
    }

    public void loadGermanDewack100Vectors(String datapath) throws IOException{

        FileReader fr = new FileReader(datapath+"translated_entities.txt");
        BufferedReader br = new BufferedReader(fr);
        String line = br.readLine();
        int entity_counter = 0;
        while (line != null) {
         
            entitiesWordtranslateWord.put(line.split("\\|")[0], line.split("\\|")[1]);
            entitiesNumtranslateWord.put(entity_counter, line.split("\\|")[1]);

         
            for (int i = 0; i < line.split("\\_").length; i++) {
                if (!vocabDE.contains(line.split("\\_")[i])) {
                    vocabDE.add(line.split("\\_")[i]);
                }
            }
            line = br.readLine();
            entity_counter++;
        }
        br.close();
        vocabDE.add("unbekannt");

        System.out.println(entity_counter + "Entity translation loaded with a vocab size of: "+vocabDE.size());

       
        fr = new FileReader(datapath+"dewac_vectors100.bin");
        br = new BufferedReader(fr);
       
        line = br.readLine();
       
        line = br.readLine();
       
        int wvCounterDe = 0;
        String word;
        while (line != null) {
            word = line.split("\\s")[0];
            if (wvCounterDe<100) {
        
            }
            if (vocabDE.contains(word)) {
      
                vocabNumWordDE.put(wvCounterDe, word);
                vocabWordNumDE.put(word, wvCounterDe);
                INDArray wv = Nd4j.create(100,1);
                for (int i = 1; i < embeddings_size+1; i++) {
                    wv.putScalar(i-1, Double.parseDouble(line.split("\\s")[i]));
                }
                worvectorWordVecDE.put(word, wv);
                worvectorNumVecDE.put(wvCounterDe,wv);
                wvCounterDe++;
            }
            line = br.readLine();
        }
        br.close();

        numOfWords = worvectorNumVecDE.size();
       

       
        vocabNumWord = vocabNumWordDE;
        vocabWordNum = vocabWordNumDE;

       
        wordVectorMaxtrixLoaded = Nd4j.zeros(100, numOfWords);
        for (int i = 0; i < numOfWords; i++) {
            wordVectorMaxtrixLoaded.putColumn(i, worvectorNumVecDE.get(i));
       
        }
        System.out.println("word vector matrix with german words is ready..."+wordVectorMaxtrixLoaded);
    }

    public INDArray createEntityVectorsByWordEmbeddings(INDArray updatedWVMatrix){
        INDArray entity_vectors = Nd4j.zeros(embeddings_size,numOfentities);
        int wv_not_found_counter =0;
        for (int i = 0; i < numOfentities; i++) {
            String entity_name=null; 
            if(german ==false){
                try {
                    entity_name = entitiesNumWord.get(i).substring(2, entitiesNumWord.get(i).lastIndexOf("_"));
                } catch (Exception e) {
                    entity_name =entitiesNumWord.get(i).substring(2);
                }
                
            }else{
                entity_name = entitiesNumtranslateWord.get(i);
            }

            if (entity_name.contains("_")) { 
                
                INDArray entityvector = Nd4j.zeros(embeddings_size, 1);
                int counterOfWordVecs = 0;
                for (int j = 0; j <entity_name.split("_").length; j++) {
                    try {
                        
                        entityvector = entityvector.add(updatedWVMatrix.getColumn(vocabWordNum.get(entity_name.split("_")[j])));
                    } catch (Exception e) {
                        
                        wv_not_found_counter++;
                        if(german ==false){
                            
                            entityvector = entityvector.add(updatedWVMatrix.getColumn(vocabWordNum.get("unknown")));
							
                        }else{
                            entityvector = entityvector.add(updatedWVMatrix.getColumn(vocabWordNum.get("unbekannt")));
                        }
                    }
                    counterOfWordVecs++;
                }
                entityvector = entityvector.div(counterOfWordVecs);
                entity_vectors.putColumn(i, entityvector);
            }else{
                
                try {
                    entity_vectors.putColumn(i, updatedWVMatrix.getColumn(vocabWordNum.get(entity_name)));
                
                
                } catch (Exception e) {
                
                    wv_not_found_counter++;
                
                    if(german ==false){
                        entity_vectors.putColumn(i, updatedWVMatrix.getColumn(vocabWordNum.get("unknown")));
				
                    }else{
                        entity_vectors.putColumn(i, updatedWVMatrix.getColumn(vocabWordNum.get("unbekannt")));
                    }
                }
            }
        }
       
        return entity_vectors;
    }

    public int entityLength(int entityIndexNum){

        try {
            return entitiesNumWord.get(entityIndexNum).split("_").length-3;
        } catch (Exception e) {
         
            System.out.println("entityIndexNum: "+entityIndexNum+" | "+entitiesNumWord.get(entityIndexNum)+" | false vocab word by entitynum: "+vocabNumWord.get(entityIndexNum));
            return 1;
        }


    }

    public ArrayList<Tripple> getDevTripples() {
        return devTripples;
    }
    public ArrayList<Tripple> getTestTripples() {
        return testTripples;
    }
    public int getNumOfWords() {
        return numOfWords;
    }

    public int getNumOfRelations() {
        return numOfRelations;
    }

    public INDArray getWordVectorMaxtrixLoaded() {
        return wordVectorMaxtrixLoaded;
    }
    public INDArray getEntitiy1IndexNumbers(ArrayList<Tripple> list){
        //number is corresponding to column in entityvectors matrix
        INDArray e1_list = Nd4j.create(list.size());
        for (int i = 0; i < list.size(); i++) {
            e1_list.putScalar(i, list.get(i).getIndex_entity1());
        }
        return e1_list;
    }
    public INDArray getEntitiy2IndexNumbers(ArrayList<Tripple> list){
        //number is corresponding to column in entityvectors matrix
        INDArray e2_list = Nd4j.create(list.size());
        for (int i = 0; i < list.size(); i++) {
            e2_list.putScalar(i, list.get(i).getIndex_entity2());
        }
        return e2_list;
    }
    public INDArray getRelIndexNumbers(ArrayList<Tripple> list){
        //number is corresponding to column in entityvectors matrix
        INDArray rel_list = Nd4j.create(list.size());
        for (int i = 0; i < list.size(); i++) {
            rel_list.putScalar(i, list.get(i).getIndex_relation());
        }
        return rel_list;
    }
    public INDArray getEntitiy3IndexNumbers(ArrayList<Tripple> list){
        //number is corresponding to column in entityvectors matrix
        INDArray e3_list = Nd4j.create(list.size());
        for (int i = 0; i < list.size(); i++) {
            e3_list.putScalar(i, list.get(i).getIndex_entity3_corrupt());
        }
        return e3_list;
    }
    public int[] getWordIndexes(int entityIndex){
        int[] wordIndexes = new int[entityLength(entityIndex)];
        if (entityLength(entityIndex)==0) {
            //System.out.println("+++++ "+entitiesNumWord.get(entityIndex) +" entityLength(entityIndex)"+entityLength(entityIndex));
            //exception for corrupt training data: entityIndexNum: 9847 | __2 |
            wordIndexes = new int[1];
        }

        // get words of entity
        String entity_name; //clear name without _  __name_ -> name
        try {
            entity_name = entitiesNumWord.get(entityIndex).substring(2, entitiesNumWord.get(entityIndex).lastIndexOf("_"));
        } catch (Exception e) {
            entity_name =entitiesNumWord.get(entityIndex).substring(2);
        }

        // get word indexes
        if (entity_name.contains("_")) { //whitespaces are _
            //Entity conains of more than one word
            for (int j = 0; j <entity_name.split("_").length; j++) {
                try {
                    wordIndexes[j] = vocabWordNum.get(entity_name.split("_")[j]);
                } catch (Exception e) {
                    //if no word vector available, use "unknown" word vector
                    wordIndexes[j] = vocabWordNum.get("unknown");
                }
            }
        }else{
            // Entity conains of only one word
            try {
                wordIndexes[0] = vocabWordNum.get(entity_name);
            } catch (Exception e) {
                // if no word vector available, use "unknown" word vector
                wordIndexes[0] = vocabWordNum.get("unknown");
            }
        }
        //System.out.println("wordIndexes: "+ wordIndexes.length + " | "+wordIndexes[0]);
        return wordIndexes;
    }



}
