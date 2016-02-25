import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class ApLearn {

    private static Map<String,String> getParamMap(String args[]) {
        HashMap<String,String> name2Value = new HashMap<String, String>();

        int idx = 0;
        while (idx < args.length) {
            String key = args[idx];
            if (!key.startsWith("--")) {
                return null;
            }
            idx++;
            if (idx == args.length) {
                return null;
            }
            name2Value.put(key, args[idx]);
            idx++;
        }
        return name2Value;

    }

    
    private static void printUsageExit() {
        System.out.println("Usage: java ApLearn --train-set [file name] --model-name [file name] --epoch-num [num] ");
    }
    /**
     * @param args
     */
    public static void main(String[] args) throws  IOException {
        Map<String,String> paramMap = getParamMap(args);
        if (paramMap.size() == 0) {
            printUsageExit();
        }
        
        System.out.println("Params: " + paramMap);
        int epochNum = paramMap.containsKey("--epoch-num") ? Integer.parseInt(paramMap.get("--epoch-num")) : 10;
        
        String fileName = paramMap.get("--train-set");
        String modelName = paramMap.get("--model-name");
        
        InputReader reader = new InputReader();
        List<Example> data = reader.readData(fileName);
       
        if (paramMap.containsKey("--std-perceptron") && Boolean.parseBoolean(paramMap.get("--std-perceptron"))) {
            StdPerceptron sp = new StdPerceptron();
            Model mod = sp.learn(data, epochNum);
            mod.save(modelName);
        } else {
        
            AveragedPerceptron ap = new AveragedPerceptron();
            Model mod = ap.learn(data, epochNum);
            mod.save(modelName);
        }
        
    }

}
