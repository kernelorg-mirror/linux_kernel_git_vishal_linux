Using arch_prctl to request AMX capabilities
============================================

Intel AMX (Advanced Matrix Extensions) is a dynamically enabled feature
that requires a process to request and obtain prior permission from the
kernel before it can be used.

The  following is an example program that obtains the necessary permissions,
sets up the sigaltstack, and then performs a simple matrix dot product.

Toolchain notes
---------------

This requires at least gcc-11.2, glibc-2.34 and binutils-2.37.

Example Program
---------------

.. code-block:: C

 
        /*
         * AMX usage example
         *
         * This performs the following high level steps:
         *
         *   1. Detect AMX tile architecture support - CPUID.0x7.0.EDX.AMX_TILE[bit 24] = 1
         *   2. Setup sigaltstack with proper size (see setup_sigaltstack())
         *   3. Request permission to use AMX tile data (see request_perm_xtile_data())
         *   4. Load data and compute dot product (see load_rand_tiledata() and mult_abc())
         */
        
        #define _GNU_SOURCE
        #include <err.h>
        #include <errno.h>
        #include <stdio.h>
        #include <string.h>
        #include <stdbool.h>
        #include <unistd.h>
        #include <x86intrin.h>
        #include <immintrin.h>
        
        #include <sys/auxv.h>
        #include <sys/mman.h>
        #include <sys/syscall.h>
        #include <sys/signal.h>
        
        #define fatal_error(msg, ...)	err(1, "[FAIL]\t" msg, ##__VA_ARGS__)
        
        #ifndef AT_MINSIGSTKSZ
        #  define AT_MINSIGSTKSZ	51
        #endif
        
        #define XFEATURE_XTILECFG	17
        #define XFEATURE_XTILEDATA	18
        #define XFEATURE_MASK_XTILECFG	(1 << XFEATURE_XTILECFG)
        #define XFEATURE_MASK_XTILEDATA	(1 << XFEATURE_XTILEDATA)
        #define XFEATURE_MASK_XTILE	(XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
        
        #define ARCH_GET_XCOMP_PERM	0x1022
        #define ARCH_REQ_XCOMP_PERM	0x1023
        
        #define TILE_M			8
        #define TILE_K			8
        #define TILE_N			8
        #define MAX_ELEMENTS		((TILE_M * TILE_K) + (TILE_K * TILE_N) + (TILE_M * TILE_N))
        #define BYTES_PER_ELEMENT	4
        
        struct tile_buffer {
        	union {
        		struct {
        			uint32_t a[TILE_M * TILE_K];
        			uint32_t b[TILE_K * TILE_N];
        			uint32_t c[TILE_M * TILE_N];
        		};
        		uint32_t bytes[0];
        	};
        };
        
        static inline void cpuid(uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx)
        {
        	asm volatile("cpuid;"
        		     : "=a" (*eax), "=b" (*ebx), "=c" (*ecx), "=d" (*edx)
        		     : "0" (*eax), "2" (*ecx));
        }
        
        #define CPUID_LEAF_XFEATURE_ENUM		0x07
        #define CPUID_LEAF_TILE_INFO			0x1d
        #define CPUID_LEAF_TMUL_INFO			0x1e
        
        #define CPUID_LEAF_XFEATURE_AMX_TILE_SHIFT	24
        #define CPUID_LEAF_XFEATURE_AMX_INT8_SHIFT	25
        
        #define CPUID_SUBLEAF_TILE_ECX_PALETTE_1	1
        
        #define CPUID_TILE_BYTES_MASK			0xffff
        #define CPUID_TILE_BYTES_PER_TILE_SHIFT		16
        #define CPUID_TILES_MAX_SHIFT			16
        #define CPUID_TILE_BYTES_PER_ROW_MASK		0xffff
        #define CPUID_TILE_ROWS_MAX_MASK		0xffff
        
        #define CPUID_TMUL_MAXK_MASK			0xff
        #define CPUID_TMUL_MAXN_MASK			0xffff
        #define CPUID_TMUL_MAXN_SHIFT			0x8
        
        #define TMM0	0
        #define TMM1	1
        #define TMM2	2
        #define TMM3	3
        #define TMM4	4
        #define TMM5	5
        #define TMM6	6
        #define TMM7	7
        
        static uint32_t max_palette;
        static uint32_t total_tile_bytes, bytes_per_tile, max_tiles;
        static uint32_t bytes_per_row, max_rows;
        static uint32_t tmul_maxk, tmul_maxn;
        
        static void amx_check_cpuid(void)
        {
        	uint32_t eax, ebx, ecx, edx;
        
        	eax = CPUID_LEAF_XFEATURE_ENUM;
        	ecx = 0;
        	cpuid(&eax, &ebx, &ecx, &edx);
        	if (!((edx >> CPUID_LEAF_XFEATURE_AMX_TILE_SHIFT) & 0x1))
        		fatal_error("CPUID: AMX Tile architecture not supported");
        	if (!((edx >> CPUID_LEAF_XFEATURE_AMX_INT8_SHIFT) & 0x1))
        		fatal_error("CPUID: AMX-INT8 operations not supported");
        
        	eax = CPUID_LEAF_TILE_INFO;
        	ecx = 0;
        	cpuid(&eax, &ebx, &ecx, &edx);
        
        	max_palette = eax;
        	printf("CPUID Tile Info leaf:\n");
        	printf("  max_palette: %u\n", max_palette);
        
        	if (!max_palette)
        		fatal_error("AMX support missing (max_palette = 0)");
        
        	eax = CPUID_LEAF_TILE_INFO;
        	ecx = CPUID_SUBLEAF_TILE_ECX_PALETTE_1;
        	cpuid(&eax, &ebx, &ecx, &edx);
        
        	total_tile_bytes = eax & CPUID_TILE_BYTES_MASK;
        	bytes_per_tile = eax >> CPUID_TILE_BYTES_PER_TILE_SHIFT;
        	bytes_per_row = ebx & CPUID_TILE_BYTES_PER_ROW_MASK;
        	max_tiles = ebx >> CPUID_TILES_MAX_SHIFT;
        	max_rows = ecx & CPUID_TILE_ROWS_MAX_MASK;
        	printf("  total_tile_bytes: %u\n", total_tile_bytes);
        	printf("  bytes_per_tile: %u\n", bytes_per_tile);
        	printf("  bytes_per_row: %u\n", bytes_per_row);
        	printf("  max_tiles: %u\n", max_tiles);
        	printf("  max_rows: %u\n", max_rows);
        
        	eax = CPUID_LEAF_TMUL_INFO;
        	ecx = 0;
        	cpuid(&eax, &ebx, &ecx, &edx);
        
        	tmul_maxk = ebx & CPUID_TMUL_MAXK_MASK;
        	tmul_maxn = (ebx >> CPUID_TMUL_MAXN_SHIFT) & CPUID_TMUL_MAXN_MASK;
        	printf("CPUID TMUL Info leaf:\n");
        	printf("  tmul_maxk: %u\n", tmul_maxk);
        	printf("  tmul_maxn: %u\n", tmul_maxn);
        }
        
        static struct tilecfg {
        	uint8_t palette;	/* byte 0 */
        	uint8_t start_row;	/* byte 1 */
        	char rsvd1[14];		/* bytes 2-15 */
        	uint16_t tile_colsb[8];	/* bytes 16-31 */
        	char rsvd2[16];		/* bytes 32-47 */
        	uint8_t tile_rows[8];	/* bytes 48-55 */
        	char rsvd3[8];		/* bytes 56-63 */
        } __attribute__((packed)) tilecfg;
        
        static void print_tilecfg(struct tilecfg *t)
        {
        	int i;
        
        	printf("TILECFG:\n");
        	printf("  palette: %d\n", t->palette);
        	printf("  start_row: %d\n", t->start_row);
        	for(i = 0; i < 8; i++)
        		printf("  tmm%d: [ %d x %d ]\n", i, t->tile_rows[i], t->tile_colsb[i]);
        }
        
        static void load_tile_config(struct tilecfg *t)
        {
        	t->palette = 1;
        	t->start_row = 0;
        
        	t->tile_rows[TMM0] = TILE_M;	/* tmm0 -> A: src1 matrix, MxK */
        	t->tile_colsb[TMM0] = TILE_K * BYTES_PER_ELEMENT;
        
        	t->tile_rows[TMM1] = TILE_K;	/* tmm1 -> B: src2 matrix, KxN */
        	t->tile_colsb[TMM1] = TILE_N * BYTES_PER_ELEMENT;
        
        	t->tile_rows[TMM2] = TILE_M;	/* tmm2 -> C: dst matrix, MxN */
        	t->tile_colsb[TMM2] = TILE_N * BYTES_PER_ELEMENT;
        
        	_tile_loadconfig(t);
        }
        
        static void get_stored_tilecfg(struct tilecfg *t)
        {
        	_tile_storeconfig(t);
        }
        
        static void set_rand_tiledata(struct tile_buffer *tbuf)
        {
        	int data;
        	int i;
        
        	/*
        	 * Ensure that 'data' is never 0.  This ensures that
        	 * the registers are never in their initial configuration
        	 * and thus never tracked as being in the init state.
        	 */
        
        	for (i = 0; i < (MAX_ELEMENTS - (TILE_M * TILE_N)); i++) {
        		data = (rand() % 0xff) | 1;
        		tbuf->bytes[i] = data;
        	}
        }
        
        static void load_rand_tiledata(struct tile_buffer *tbuf)
        {
        	set_rand_tiledata(tbuf);
        
        	_tile_release();
        	printf("TILERELEASE Done\n");
        
        	load_tile_config(&tilecfg);
        	printf("LDTILECFG Done\n");
        
        	get_stored_tilecfg(&tilecfg);
        	print_tilecfg(&tilecfg);
        
        	_tile_loadd(TMM0, &tbuf->a[0], TILE_K * BYTES_PER_ELEMENT);
        	printf("TILELOADD tmm0 Done\n");
        	_tile_loadd(TMM1, &tbuf->b[0], TILE_N * BYTES_PER_ELEMENT);
        	printf("TILELOADD tmm1 Done\n");
        	_tile_loadd(TMM2, &tbuf->c[0], TILE_N * BYTES_PER_ELEMENT);
        	printf("TILELOADD tmm2 Done\n");
        }
        
        static void mult_abc(struct tile_buffer *tbuf)
        {
        	_tile_dpbuud(TMM2, TMM0, TMM1);
        	printf("TDPBUUD (tmm2 += tmm0 . tmm1) Done\n");
        
        	_tile_stored(TMM2, &tbuf->c[0], TILE_N * BYTES_PER_ELEMENT);
        	printf("TILESTORED (tmm2-> 'C') Done\n");
        }
        
        static void request_perm_xtile_data()
        {
        	unsigned long bitmask;
        	long rc;
        
        	rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
        	if (rc)
        		fatal_error("XTILE_DATA request failed: %ld", rc);
        
        	rc = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
        	if (rc)
        		fatal_error("prctl(ARCH_GET_XCOMP_PERM) error: %ld", rc);
        
        	if (bitmask & XFEATURE_MASK_XTILE)
        		printf("ARCH_REQ_XCOMP_PERM XTILE_DATA successful.\n");
        }
        
        static void setup_sigaltstack()
        {
        	unsigned long minsigstksz, new_size;
        	void *altstack;
        	stack_t ss;
        	int rc;
        
        	minsigstksz = getauxval(AT_MINSIGSTKSZ);
        	printf("AT_MINSIGSTKSZ = %lu\n", minsigstksz);
        	/*
        	 * getauxval() itself can return 0 for failure or
        	 * success.  But, in this case, AT_MINSIGSTKSZ
        	 * will always return a >=0 value if implemented.
        	 * Just check for 0.
        	 */
        	if (minsigstksz == 0)
        		fatal_error("no support for AT_MINSIGSTKSZ");
        
        	new_size = minsigstksz * 2;
        	altstack =  mmap(NULL, new_size, PROT_READ | PROT_WRITE,
        			MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
        	if (altstack == MAP_FAILED)
        		fatal_error("mmap() for altstack");
        
        	memset(&ss, 0, sizeof(ss));
        	ss.ss_size = new_size;
        	ss.ss_sp = altstack;
        
        	rc = sigaltstack(&ss, NULL);
        	if (rc)
        		fatal_error("sigaltstack failed: %d", rc);
        
        }
        
        static void print_abc(struct tile_buffer *tbuf)
        {
        	int i, j;
        
        	/* printf("Raw Buffer\n [");
        	for (i = 0; i < MAX_ELEMENTS; i++)
        		printf(" %u", tbuf->bytes[i]);
        	printf(" ]\n\n");
        	*/
        
        	printf("Matrix A:\n");
        	for (i = 0; i < TILE_M; i++) {
        		printf(" [");
        		for (j = 0; j < TILE_K; j++)
        			printf(" %03u", tbuf->a[(i * TILE_K) + j]);
        		printf(" ]\n");
        	}
        	printf("\n");
        
        	printf("Matrix B:\n");
        	for (i = 0; i < TILE_K; i++) {
        		printf(" [");
        		for (j = 0; j < TILE_N; j++)
        			printf(" %03u", tbuf->b[(i * TILE_N) + j]);
        		printf(" ]\n");
        	}
        	printf("\n");
        
        	printf("Matrix C:\n");
        	for (i = 0; i < TILE_M; i++) {
        		printf(" [");
        		for (j = 0; j < TILE_N; j++)
        			printf(" %06u", tbuf->c[(i * TILE_N) + j]);
        		printf(" ]\n");
        	}
        	printf("\n");
        }
        
        int main(void)
        {
        	struct tile_buffer *tile;
        
        	amx_check_cpuid();
        	tile = aligned_alloc(64, MAX_ELEMENTS * BYTES_PER_ELEMENT);
        	if (!tile)
        		fatal_error("failed to allocate tile");
        
        	setup_sigaltstack();
        
        	/* Load tile configuration and tile data for matrices */
        	request_perm_xtile_data();
        	load_rand_tiledata(tile);
        
        	printf("\nA, B, C matrices before dot product:\n");
        	print_abc(tile);
        
        	/* compute the dot product, store result in the memory for 'C' */
        	mult_abc(tile);
        
        	/* print multiplication result */
        	printf("\nA, B, C matrices after dot product (C = A . B):\n");
        	print_abc(tile);
        
        	free(tile);
        	printf("All done\n");
        	return 0;
        }
