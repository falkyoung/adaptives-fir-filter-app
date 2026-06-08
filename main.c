#include <stdio.h>
#include <stdlib.h>

#define P 16          // 16 Koeffizienten
#define MU 0.02       // Lernrate

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Aufruf: %s <samples.txt> <output.csv>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "r");
    if (!in) { perror("Eingabedatei"); return 1; }
    FILE *out = fopen(argv[2], "w");
    if (!out) { perror("Ausgabedatei"); fclose(in); return 1; }

    double a[P] = {0.0};         // Filterkoeffizienten (starten bei 0)
    double x_history[P] = {0.0}; // Verzögerungsleitung (Vergangenheit)

    fprintf(out, "t,x,x_hat,error");
    for (int k = 0; k < P; k++) fprintf(out, ",a%d", k);
    fprintf(out, "\n");

    double x_current;
    long t = 0;

    // Adaptives FIR ab hier
    while (fscanf(in, "%lf", &x_current) == 1) {

        // Phase 1: Schätzung
        double x_dach = 0.0;
        for (int k = 0; k < P; k++) {
            x_dach += a[k] * x_history[k];
        }

        // Fehler berechnen
        double error = x_current - x_dach;

        // Phase 2: Koeffizienten-Anpassung (Gradientenabstieg / LMS)
        // Abgeleitet ergibt sich: Delta_a = MU * error * x_past
        for (int k = 0; k < P; k++) {
            a[k] += MU * error * x_history[k];
        }

        // (für Plots)
        fprintf(out, "%ld,%f,%f,%f", t, x_current, x_dach, error);
        for (int k = 0; k < P; k++) fprintf(out, ",%f", a[k]);
        fprintf(out, "\n");

        // Verzögerungsleitung aktualisieren
        for (int k = P - 1; k > 0; k--) {
            x_history[k] = x_history[k - 1];
        }
        x_history[0] = x_current;
        t++;
    }

    fclose(in);
    fclose(out);
}
