#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Aufruf: %s <samples.txt> <output.csv> <MU> <d1> [d2 ...]\n",
                argv[0]);
        return 1;
    }

    double mu = atof(argv[3]);
    int P = argc - 4;                 // so viele Taps wie Verzoegerungen

    int *delay = malloc(P * sizeof(int));
    int max_delay = 0;
    for (int k = 0; k < P; k++) {
        delay[k] = atoi(argv[4 + k]);
        if (delay[k] > max_delay) max_delay = delay[k];
    }

    FILE *in = fopen(argv[1], "r");
    if (!in) { perror("Eingabedatei"); return 1; }
    FILE *out = fopen(argv[2], "w");
    if (!out) { perror("Ausgabedatei"); fclose(in); return 1; }

    double *a = calloc(P, sizeof(double));            // Koeffizienten (start 0)
    double *hist = calloc(max_delay, sizeof(double)); // Vergangenheit:
                                                      // hist[d-1] = x(t-d)

    fprintf(out, "t,x,x_hat,error");
    for (int k = 0; k < P; k++) fprintf(out, ",a%d", k);
    fprintf(out, "\n");

    double x_current;
    long t = 0;

    while (fscanf(in, "%lf", &x_current) == 1) {

        // Phase 1: Schaetzung aus den gewaehlten Vergangenheitswerten
        double x_dach = 0.0;
        for (int k = 0; k < P; k++) {
            x_dach += a[k] * hist[delay[k] - 1];
        }

        // Fehler
        double error = x_current - x_dach;

        // Phase 2: Koeffizienten anpassen (LMS: Delta_a = MU * error * x_past)
        for (int k = 0; k < P; k++) {
            a[k] += mu * error * hist[delay[k] - 1];
        }

        // (fuer Plots)
        fprintf(out, "%ld,%f,%f,%f", t, x_current, x_dach, error);
        for (int k = 0; k < P; k++) fprintf(out, ",%f", a[k]);
        fprintf(out, "\n");

        // Verzoegerungsleitung weiterschieben, neuen Wert vorne einsetzen
        for (int i = max_delay - 1; i > 0; i--) {
            hist[i] = hist[i - 1];
        }
        hist[0] = x_current;
        t++;
    }

    free(delay);
    free(a);
    free(hist);
    fclose(in);
    fclose(out);
    return 0;
}
