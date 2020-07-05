#Credits to Lin Zhouhan(@hantek) for the complete visualization code
import random, os, numpy, scipy
from codecs import open

class Visualization(object):
    def __init__(self, savedir, select_epoch, num_of_reviews):
        super(Visualization, self).__init__()
        self.savedir = savedir + "/VisualizeAttn/epoch_{}/htmlhm".format(select_epoch)
        os.makedirs(self.savedir, exist_ok=True)

        for i in range(num_of_reviews):
            os.makedirs(self.savedir + "/{}".format(i), exist_ok=True)
    
    def wdIndex2sentences(self, word_index, idx2wd, weights):
        
        words = [idx2wd[index] for index in word_index if idx2wd[index] != 'PAD']
        weights = [weights[index] for index, idx in enumerate(word_index) if idx2wd[idx] != 'PAD']
        sentence = [" ".join(words)]
        return sentence, [weights]

    def createHTML(self, texts, weights, reviews_ctr, fname):
        """
        Creates a html file with text heat.
        weights: attention weights for visualizing
        texts: text on which attention weights are to be visualized
        """
        
        fileName = R'{}/{}.html'.format(self.savedir + "/{}".
            format(reviews_ctr), fname)
        
        fOut = open(fileName, "w", encoding="utf-8")
        part1 = """
        <html lang="en">
        <head>
        <meta http-equiv="content-type" content="text/html; charset=utf-8">
        <style>
        body {
        font-family: Sans-Serif;
        }
        </style>
        </head>
        <body>
        <h3>
        Heatmaps
        </h3>
        </body>
        <script>
        """
        part2 = """
        var color = "255,0,0";
        var ngram_length = 3;
        var half_ngram = 1;
        for (var k=0; k < any_text.length; k++) {
        var tokens = any_text[k].split(" ");
        var intensity = new Array(tokens.length);
        var max_intensity = Number.MIN_SAFE_INTEGER;
        var min_intensity = Number.MAX_SAFE_INTEGER;
        for (var i = 0; i < intensity.length; i++) {
        intensity[i] = 0.0;
        // 新增規則 weights大於0才作畫
        for (var j = -half_ngram; j < ngram_length-half_ngram; j++) {
        if (i+j < intensity.length && i+j > -1 && trigram_weights[k][i] > 0) {
        intensity[i] += trigram_weights[k][i + j];
        }
        }
        if (i == 0 || i == intensity.length-1) {
        intensity[i] /= 2.0;
        } else {
        intensity[i] /= 3.0;
        }
        if (intensity[i] > max_intensity) {
        max_intensity = intensity[i];
        }
        if (intensity[i] < min_intensity) {
        min_intensity = intensity[i];
        }
        }
        var denominator = max_intensity - min_intensity;
        for (var i = 0; i < intensity.length; i++) {
        intensity[i] = (intensity[i] - min_intensity) / denominator;
        }
        if (k%2 == 0) {
        var heat_text = "<p><br><b>Example {fname}:</b><br>";
        } else {
        var heat_text = "<b>Example:</b><br>";
        }
        var space = "";
        for (var i = 0; i < tokens.length; i++) {
        heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";
        if (space == "") {
        space = " ";
        }
        }
        //heat_text += "<p>";
        document.body.innerHTML += heat_text;
        }
        </script>
        </html>"""

        putQuote = lambda x: "\"%s\""%x
        textsString = "var any_text = [%s];\n"%(",".join(map(putQuote, texts)))
        weightsString = "var trigram_weights = [%s];\n"%(",".join(map(str,weights)))
        fOut.write(part1)
        fOut.write(textsString)
        fOut.write(weightsString)
        fOut.write(part2)
        fOut.close()
    
        return

if __name__ == "__main__":
    """
    Generate for testing.
    """
    # weights = [[0.00451355054974556,0.007237346842885017,0.009158979170024395,0.013832096010446548,0.015404355712234974,0.020429469645023346,0.028239678591489792,0.032536886632442474,0.030467107892036438,0.025441039353609085,0.028424259275197983,0.034465596079826355,0.03796358406543732,0.0362275131046772,0.029924951493740082,0.02914220094680786,0.03236296400427818,0.03258327767252922,0.024766698479652405,0.026832353323698044,0.027459558099508286,0.024560749530792236,0.031061017885804176,0.024571212008595467,0.0222282987087965,0.020525561645627022,0.019024724140763283,0.016417646780610085,0.016872277483344078,0.027293210849165916,0.019125068560242653,0.017782215029001236,0.017704017460346222,0.015914319083094597,0.013810127042233944,0.013248131610453129,0.008315317332744598]]
    # words = ['dockers','men','mm','feather','edge','belt','good','buy','price','wide','belt','grace','pants','good','hold','hips','good','looks','read','carhartt','belt','review','ordered','belt','also','one','size','larger','waiste','able','slide','end','belt','loop','buckle','recommend','product']
    # texts = [" ".join(words)]    

    # fileName = "HNAE/visualization/testing.html"
    # createHTML(texts, weights, fileName)

    pass

