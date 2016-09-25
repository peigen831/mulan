/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package mulan.examples;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.SerializationHelper;
import weka.core.Utils;

/**
 * This examples shows how you can retrieve the predictions of a model on
 * unlabeled data. Unlabeled multi-label datasets should have the same
 * structure as the training data. The actual values of the labels could be
 * either unspecified (set to symbol ?), or randomly set to 0/1.
 *
 * @author Grigorios Tsoumakas
 * @version 2010.12.15
 */
public class GettingPredictionsOnUnlabeledData {

    /**
     * Executes this example
     *
     * @param args command-line arguments -arff, -xml and -unlabeled
     */
    public static void main(String[] args) {
    	//-arff data\tt_training.arff -xml data\tt.xml -unlabeled data\tt_testing.arff
        try {
            String arffFilename = Utils.getOption("arff", args);
            String xmlFilename = Utils.getOption("xml", args);
            System.out.println("Loading the training data set...");
            MultiLabelInstances trainingData = new MultiLabelInstances(arffFilename, xmlFilename);

            RAkEL model = new RAkEL(new LabelPowerset(new J48()));

            String modelName = "ttmodel.dat";
            
//            Store and build model 
            System.out.println("Building the model...");
            model.build(trainingData);
            SerializationHelper.write(modelName, model);
            
            //Load model
//            System.out.println("Loading model...");
//            RAkEL learner2 = (RAkEL) SerializationHelper.read(modelName);
            
            model.setSizeOfSubset(2);

            String unlabeledDataFilename = Utils.getOption("unlabeled", args);
            System.out.println("Loading the unlabeled data set...");
            MultiLabelInstances unlabeledData = new MultiLabelInstances(unlabeledDataFilename, xmlFilename);

            int numInstances = unlabeledData.getNumInstances();
            System.out.println("Number instance: "+ numInstances);
            for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
                Instance instance = unlabeledData.getDataSet().instance(instanceIndex);
                MultiLabelOutput output = model.makePrediction(instance);
                System.out.println("Predicted bipartion: " + Arrays.toString(output.getBipartition()));
                System.out.println("Confidence: " + Arrays.toString(output.getConfidences()));
                System.out.println("Ranking: " + Arrays.toString(output.getRanking()));
                System.out.println("Pvalue: " + output.getPvalues());
//                if (output.hasBipartition()) {
//                    String bipartion = Arrays.toString(output.getBipartition());
//                }
            }
        } catch (InvalidDataFormatException e) {
            System.err.println(e.getMessage());
        } catch (Exception ex) {
            Logger.getLogger(GettingPredictionsOnUnlabeledData.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}