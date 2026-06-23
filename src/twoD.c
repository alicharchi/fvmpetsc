#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>

static char help[] =
    "Solves 2D steady heat equation using finite volume method.\n";

PetscErrorCode WriteResults(Vec T);

int main(int argc, char **args)
{
    PetscCall(PetscInitialize(&argc, &args, NULL, help));

    DM da;
    Mat A;
    Vec T, b;
    KSP ksp;
    PC pc;

    /* Grid */
    PetscInt nx = 100, ny = 100;

    /* Domain */
    PetscReal Lx = 1.0, Ly = 1.0;

    /* Physical properties */
    PetscReal alpha = 1.0;

    /* Boundary conditions */
    PetscReal bcW = 0.0;
    PetscReal bcE = 100.0;
    PetscReal bcS = 0.0;
    PetscReal bcN = 0.0;
    
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-Lx", &Lx, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-Ly", &Ly, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &alpha, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-bcW", &bcW, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-bcE", &bcE, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-bcN", &bcN, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-bcS", &bcS, NULL));    

    /* Create structured grid */
    PetscCall(
        DMDACreate2d(
            PETSC_COMM_WORLD,
            DM_BOUNDARY_NONE,
            DM_BOUNDARY_NONE,
            DMDA_STENCIL_STAR,
            nx,
            ny,
            PETSC_DECIDE,
            PETSC_DECIDE,
            1, /* one dof = temperature */
            1, /* stencil width */
            NULL,
            NULL,
            &da));
   
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMDASetUniformCoordinates(da, 0.0, Lx, 0.0, Ly, 0.0, 0.0));

    PetscCall(DMDAGetInfo(da, NULL, &nx, &ny, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));

    PetscReal dx = Lx / nx;
    PetscReal dy = Ly / ny;

   /* Face areas */
    PetscReal Aw = dy;
    PetscReal Ae = dy;
    PetscReal As = dx;
    PetscReal An = dx;

    /* Create matrix and vectors */
    PetscCall(DMCreateMatrix(da, &A));
    PetscCall(DMCreateGlobalVector(da, &T));
    PetscCall(VecDuplicate(T, &b));

   PetscScalar **barr;
   PetscCall(DMDAVecGetArray(da, b, &barr));

    PetscCall(PetscObjectSetName((PetscObject)T, "Temperature"));

    /* Get local ownership region  */
    PetscInt xs, ys, xm, ym;

    PetscCall(
        DMDAGetCorners(
            da,
            &xs,
            &ys,
            NULL,
            &xm,
            &ym,
            NULL));

    /* Assemble FVM matrix */
    for (PetscInt j = ys; j < ys + ym; j++)
    {
        for (PetscInt i = xs; i < xs + xm; i++)
        {
            PetscReal aW, aE, aS, aN, aP;
            PetscReal Sp = 0.0;
            PetscReal Su = 0.0;

            PetscScalar val[5];
            MatStencil row, col[5];

            PetscInt k = 0;

            if (i == 0)
            {
                aW = 0.0;
                aE = alpha * Ae / dx;

                Sp += -2.0 * alpha * Aw / dx;
                Su += (2.0 * alpha * Aw / dx) * bcW;
            }
            else if (i == nx - 1)
            {
                aW = alpha * Aw / dx;
                aE = 0.0;

                Sp += -2.0 * alpha * Ae / dx;
                Su += (2.0 * alpha * Ae / dx) * bcE;
            }
            else
            {
                aW = alpha * Aw / dx;
                aE = alpha * Ae / dx;
            }

            if (j == 0)
            {
                aS = 0.0;
                aN = alpha * An / dy;

                Sp += -2.0 * alpha * As / dy;
                Su += (2.0 * alpha * As / dy) * bcS;
            }
            else if (j == ny - 1)
            {
                aS = alpha * As / dy;
                aN = 0.0;

                Sp += -2.0 * alpha * An / dy;
                Su += (2.0 * alpha * An / dy) * bcN;
            }
            else
            {
                aS = alpha * As / dy;
                aN = alpha * An / dy;
            }

            aP = aW + aE + aS + aN - Sp;

            row.i = i;
            row.j = j;

            col[k].i = i;
            col[k].j = j;
            val[k] = aP;
            k++;

            if (i > 0)
            {
                col[k].i = i - 1;
                col[k].j = j;
                val[k] = -aW;
                k++;
            }

            if (i < nx - 1)
            {
                col[k].i = i + 1;
                col[k].j = j;
                val[k] = -aE;
                k++;
            }

            if (j > 0)
            {
                col[k].i = i;
                col[k].j = j - 1;
                val[k] = -aS;
                k++;
            }

            if (j < ny - 1)
            {
                col[k].i = i;
                col[k].j = j + 1;
                val[k] = -aN;
                k++;
            }

            PetscCall(
                MatSetValuesStencil(
                    A,
                    1,
                    &row,
                    k,
                    col,
                    val,
                    INSERT_VALUES));

            barr[j][i] = Su;
      }
    }

    PetscCall(DMDAVecRestoreArray(da, b, &barr));

    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    /* Solve */
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));

    /* Set default solver and pc */
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCGAMG));

    /* Set provided solver/pc options, if provided by user */
    PetscCall(KSPSetFromOptions(ksp));

    /* Call the solver */
    PetscCall(KSPSolve(ksp, b, T));    

    /* Output directly for ParaView */
    PetscCall(WriteResults(T));

    /* Cleanup */
    PetscCall(VecDestroy(&T));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(DMDestroy(&da));

    PetscCall(PetscFinalize());

    return 0;
}

PetscErrorCode WriteResults(Vec T)
{
    PetscViewer viewer;

    PetscCall(
        PetscViewerVTKOpen(
            PETSC_COMM_WORLD,
            "temperature.vts",
            FILE_MODE_WRITE,
            &viewer));

    PetscCall(VecView(T, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    return PETSC_SUCCESS;
}