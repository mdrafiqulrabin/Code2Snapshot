import com.github.javaparser.JavaToken;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;

import javax.json.Json;
import javax.json.JsonArrayBuilder;
import javax.json.JsonObject;
import javax.json.JsonObjectBuilder;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

/**
 * Input: path of java program
 * Output: json of java tokens
 */
public class Main {

    public static void main(String[] args) {
        System.out.print(getJavaTokens(args[0]));
    }

    static String getJavaTokens(String filepath) {
        try {
            JsonObjectBuilder jsonObjBuilder = Json.createObjectBuilder();
            jsonObjBuilder.add("path", filepath);
            MethodDeclaration methodDec = getMethodDeclaration(filepath, null);
            jsonObjBuilder.add("method", methodDec.getName().asString());
            MethodDeclaration methodMark = getMethodDeclaration(null,
                    methodDec.setName("METHOD_NAME").toString());
            JsonArrayBuilder allTokens = getAllTokens(methodMark);
            jsonObjBuilder.add("tokens", allTokens);
            JsonObject jsonObj = jsonObjBuilder.build();
            return jsonObj.toString();
        } catch (Exception ignore){
            return "";
        }
    }

    static JsonArrayBuilder getAllTokens (MethodDeclaration md) {
        JsonArrayBuilder allTokens = Json.createArrayBuilder();
        if (md.getTokenRange().isPresent()) {
            md.getTokenRange().get().forEach(token -> {
                        if (token.getKind() >= JavaToken.Kind.COMMENT_CONTENT.getKind()) {
                            JsonArrayBuilder curToken = Json.createArrayBuilder();
                            String tokenKind = JavaToken.Kind.valueOf(token.getKind()).toString();
                            curToken.add(token.getText());
                            curToken.add(tokenKind);
                            allTokens.add(curToken);
                        }
                    }
            );
        }
        return allTokens;
    }

    static MethodDeclaration getMethodDeclaration(String filepath, String txtCode) throws IOException {
            StaticJavaParser.getConfiguration().setAttributeComments(false);
            if (txtCode == null) {
                File javaFile = new File(filepath);
                txtCode = new String(Files.readAllBytes(javaFile.toPath()));
            }
            if(!txtCode.startsWith("class")) {
                txtCode = "class T { \n" + txtCode + "\n}";
            }
            CompilationUnit cu = StaticJavaParser.parse(txtCode);
            return cu.findAll(MethodDeclaration.class).get(0);
    }

}