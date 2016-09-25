package explore;

import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.Measure;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class MulanExp1 {
	
	/**
	 * Evaluates a {@link MultiLabelLearner} on given test data set using specified evaluation measures
	 *
	 * @param learner the learner to be evaluated via cross-validation
	 * @param data the data set for cross-validation
	 * @param measures the evaluation measures to compute
	 * @return an Evaluation object
	 * @throws IllegalArgumentException if an input parameter is null
	 * @throws Exception
	 */
//	public Evaluation evaluate(MultiLabelLearner learner, MultiLabelInstances data, List<Measure> measures) throws IllegalArgumentException, Exception {
//	    checkLearner(learner);
//	    checkData(data);
//	    checkMeasures(measures);

	    // reset measures
//	    for (Measure m : measures) {
//	        m.reset();
//	    }
//
//	    int numLabels = data.getNumLabels();
//	    int[] labelIndices = data.getLabelIndices();
//	    boolean[] trueLabels = new boolean[numLabels];
//	    Set<Measure> failed = new HashSet<Measure>();
//	    Instances testData = data.getDataSet();
//	    int numInstances = testData.numInstances();
//	    
//	    for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
//	        Instance instance = testData.instance(instanceIndex);
//	        
//	        if (data.hasMissingLabels(instance)) {
//	            continue;
//	        }
//	        
//	        MultiLabelOutput output = learner.makePrediction(instance);
//	        trueLabels = getTrueLabels(instance, numLabels, labelIndices);
//	        Iterator<Measure> it = measures.iterator();
//	        while (it.hasNext()) {
//	            Measure m = it.next();
//	            if (!failed.contains(m)) {
//	                try {
//	                    m.update(output, trueLabels);
//	                } catch (Exception ex) {
//	                    failed.add(m);
//	                }
//	            }
//	        }
//	    }
//
//	    return new Evaluation(measures);
//	}

    public static void main(String[] args) throws Exception {
    	//-arff data\emotions.arff -xml data\emotions.xml
        String arffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml

        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);

        RAkEL learner1 = new RAkEL(new LabelPowerset(new J48()));
//        MLkNN learner2 = new MLkNN();

        Evaluator eval = new Evaluator();
        MultipleEvaluation results;

        int numFolds = 2;
        results = eval.crossValidate(learner1, dataset, numFolds);
        System.out.println(results);
//        results = eval.crossValidate(learner2, dataset, numFolds);
//        System.out.println(results);
    }
}
