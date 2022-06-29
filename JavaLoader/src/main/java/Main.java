import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;

import java.io.File;
import java.nio.file.Files;

/**
 * Input: path of java program
 * Output: body of java program
 */
public class Main {

    public static void main(String[] args) {
        File inputFile = new File(args[0]);
        CompilationUnit root = getParseUnit(inputFile);
        System.out.print(loadJavaMethod(root));
    }

    static String loadJavaMethod(CompilationUnit root) {
        try {
            MethodDeclaration methodDec = root.findAll(MethodDeclaration.class).get(0);
            return methodDec.setName("METHOD_NAME").toString();
        } catch (Exception ignore){}
        return "";
    }

    static CompilationUnit getParseUnit(File javaFile) {
        CompilationUnit root = null;
        try {
            // remove comments
            StaticJavaParser.getConfiguration().setAttributeComments(false);

            // parse code
            String txtCode = new String(Files.readAllBytes(javaFile.toPath()));
            if(!txtCode.startsWith("class")) txtCode = "class T { \n" + txtCode + "\n}";
            root = StaticJavaParser.parse(txtCode);
        } catch (Exception ignore) {}
        return root;
    }

}