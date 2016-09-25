package explore;


import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.examples.StoringAndLoadingModels;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class MulanExp2 {
	
	/**
     * Executes this example
     *
     * @param args command-line arguments -path, -filestem and -percentage 
     * (training set), e.g. -path dataset/ -filestem emotions -percentage 67
     */
	public void TrainAndTest(String[] args){
		try {
            String path = Utils.getOption("path", args); 
            String filestem = Utils.getOption("filestem", args); 
            String percentage = Utils.getOption("percentage", args);  

            System.out.println("Loading the dataset");
            MultiLabelInstances mlDataSet = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");

            // split the data set into train and test
            Instances dataSet = mlDataSet.getDataSet();
            RemovePercentage rmvp = new RemovePercentage();
            rmvp.setInvertSelection(true);
            rmvp.setPercentage(Double.parseDouble(percentage));
            rmvp.setInputFormat(dataSet);
            Instances trainDataSet = Filter.useFilter(dataSet, rmvp);

            rmvp = new RemovePercentage();
            rmvp.setPercentage(Double.parseDouble(percentage));
            rmvp.setInputFormat(dataSet);
            Instances testDataSet = Filter.useFilter(dataSet, rmvp);

            MultiLabelInstances train = new MultiLabelInstances(trainDataSet, path + filestem + ".xml");
            MultiLabelInstances test = new MultiLabelInstances(testDataSet, path + filestem + ".xml");

            Evaluator eval = new Evaluator();
            Evaluation results;

            Classifier brClassifier = new NaiveBayes();
            BinaryRelevance br = new BinaryRelevance(brClassifier);
            br.setDebug(true);
            br.build(train);
            results = eval.evaluate(br, test, train);
            System.out.println(results);
        } catch (Exception e) {
            e.printStackTrace();
        }
	}
	
	//-train data\tt_training.arff -test data\tt_testing.arff -labels data\tt.xml -model model.dat
	public void storeModel(String trainPath, String labelPath, String modelPath){
        try {
//            String trainingDataFilename = Utils.getOption("train", args);
//            String labelsFilename = Utils.getOption("labels", args);
//            String modelFilename = Utils.getOption("model", args);
        	String trainingDataFilename = trainPath;
        	String labelsFilename = labelPath;
        	
            System.out.println("Loading the training data set...");
            MultiLabelInstances trainingData = new MultiLabelInstances(trainingDataFilename, labelsFilename);
            
            BinaryRelevance learner1 = new BinaryRelevance(new J48());

            String modelFilename = modelPath;
            System.out.println("Building the model...");
            learner1.build(trainingData);

            System.out.println("Storing the model...");
            SerializationHelper.write(modelFilename, learner1);

        } catch (Exception ex) {
            Logger.getLogger(StoringAndLoadingModels.class.getName()).log(Level.SEVERE, null, ex);
        }
	}
	
	public Object loadModel(String modelPath){
//		String modelFilename = Utils.getOption("model", args);
		String modelFilename = modelPath; 
		
		try{
			System.out.println("Loading the model...");
//	        BinaryRelevance learner2;
//	        learner2 = (BinaryRelevance) SerializationHelper.read(modelFilename);
			Object object = SerializationHelper.read(modelFilename);
			return object;
		} catch (Exception ex) {
            Logger.getLogger(StoringAndLoadingModels.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }
	}
	
	public void evaluate(Object learnerObject, String trainPath, String testPath, String labelPath){
		String testingDataFilename = testPath;
		String labelsFilename = labelPath;
		String trainingDataFilename = trainPath;
		MultiLabelLearnerBase learner = null;
		
		try{
			System.out.println("Loading the training data set...");
	        MultiLabelInstances trainingData = new MultiLabelInstances(trainingDataFilename, labelsFilename);
	        
	        System.out.println("Loading the testing data set...");
	        MultiLabelInstances testingData = new MultiLabelInstances(testingDataFilename, labelsFilename);
	        
			if(learnerObject instanceof BinaryRelevance)
				learner = (BinaryRelevance)learnerObject;
			
			int numInstances = testingData.getNumInstances();
	        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
	            Instance instance = testingData.getDataSet().instance(instanceIndex);
	            MultiLabelOutput output = learner.makePrediction(instance);
//				System.out.println("Predicted bipartion: " + Arrays.toString(output.getBipartition()));
	            System.out.println("Confidence: " + Arrays.toString(output.getConfidences()));
	            System.out.println("Ranking: " + Arrays.toString(output.getRanking()));
	        }
	        
	        Evaluator evaluator = new Evaluator();
	        Evaluation evaluation;
	        evaluation = evaluator.evaluate(learner, testingData, trainingData);
	        System.out.println(evaluation);
		} catch (Exception ex) {
            Logger.getLogger(StoringAndLoadingModels.class.getName()).log(Level.SEVERE, null, ex);
        }
	}

    public static void main(String[] args) throws Exception {
    	MulanExp2 mulan = new MulanExp2();
    	String trainPath = "data\\tt_training.arff";
    	String testPath = "data\\tt_testing.arff";
    	String labelPath = "data\\tt.xml";
    	String modelPath = "model.dat";

    	mulan.storeModel(trainPath, labelPath, modelPath);
    	Object learnerObject = mulan.loadModel(modelPath);
    	mulan.evaluate(learnerObject, trainPath, testPath, labelPath);
    	
    }
}
