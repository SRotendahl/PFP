/*
 * Headers
*/

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


/*
 * Initialisation
*/

int futhark_get_num_sizes(void);
const char *futhark_get_size_name(int);
const char *futhark_get_size_class(int);
struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_add_build_option(struct futhark_context_config *cfg,
                                             const char *opt);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_profiling(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s);
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s);
void
futhark_context_config_select_device_interactively(struct futhark_context_config *cfg);
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path);
void
futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                         const char *path);
void futhark_context_config_dump_binary_to(struct futhark_context_config *cfg,
                                           const char *path);
void futhark_context_config_load_binary_from(struct futhark_context_config *cfg,
                                             const char *path);
void
futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                              int size);
void
futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                              int num);
void
futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                             int num);
void
futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                             int num);
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
struct futhark_context
*futhark_context_new_with_command_queue(struct futhark_context_config *cfg,
                                        cl_command_queue queue);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
cl_command_queue futhark_context_get_command_queue(struct futhark_context *ctx);

/*
 * Arrays
*/

struct futhark_i32_1d ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx,
                                          int32_t *data, int64_t dim0);
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx,
                                              cl_mem data, int offset,
                                              int64_t dim0);
int futhark_free_i32_1d(struct futhark_context *ctx,
                        struct futhark_i32_1d *arr);
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data);
cl_mem futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                 struct futhark_i32_1d *arr);
int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                              struct futhark_i32_1d *arr);

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_i32_1d **out0,
                       struct futhark_i32_1d **out1);

/*
 * Miscellaneous
*/

void futhark_debugging_report(struct futhark_context *ctx);
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
// Start of panic.h.

#include <stdarg.h>

static const char *fut_progname;

static void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* For generating arbitrary-sized error messages.  It is the callers
   responsibility to free the buffer at some point. */
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); /* Must re-init. */
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}

// End of panic.h.

// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

// End of timing.h.

#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
// Start of values.h.

//// Text I/O

typedef int (*writer)(FILE*, void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces() {
  int c;
  do {
    c = getchar();
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(char *buf, int bufsize) {
 start:
  skipspaces();

  int i = 0;
  while (i < bufsize) {
    int c = getchar();
    buf[i] = c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getchar());
      goto start;
    } else if (!constituent(c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, stdin);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(char *buf, int bufsize, const char* expected) {
  next_token(buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    reader->n_elems_space * reader->elem_size);
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(char *buf, int bufsize,
                                struct array_reader *reader, int dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc(dims,sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc(dims,sizeof(int64_t));

  while (1) {
    next_token(buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  shape[0] = 0;

  if (!next_token_is(buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 1; i < dims; i++) {
    if (!next_token_is(buf, bufsize, "[")) {
      return 1;
    }

    next_token(buf, bufsize);

    if (sscanf(buf, "%d", &shape[i]) != 1) {
      return 1;
    }

    if (!next_token_is(buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(buf, bufsize, ")")) {
    return 1;
  }

  return 0;
}

static int read_str_array(int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, elem_size*reader.n_elems_space);
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNi8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNu8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

static void flip_bytes(int elem_size, unsigned char *elem) {
  for (int j=0; j<elem_size/2; j++) {
    unsigned char head = elem[j];
    int tail_index = elem_size-1-j;
    elem[j] = elem[tail_index];
    elem[tail_index] = head;
  }
}

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum();
  if (bin_type != expected_type) {
    panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum();
  if (expected_type != bin_primtype) {
    panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  uint64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    uint64_t bin_shape;
    ret = fread(&bin_shape, sizeof(bin_shape), 1, stdin);
    if (ret != 1) {
      panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i);
    }
    if (IS_BIG_ENDIAN) {
      flip_bytes(sizeof(bin_shape), (unsigned char*) &bin_shape);
    }
    elem_count *= bin_shape;
    shape[i] = (int64_t) bin_shape;
  }

  size_t elem_size = expected_type->size;
  void* tmp = realloc(*data, elem_count * elem_size);
  if (tmp == NULL) {
    panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  size_t num_elems_read = fread(*data, elem_size, elem_count, stdin);
  if (num_elems_read != elem_count) {
    panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    flip_bytes(elem_size, (unsigned char*) *data);
  }

  return 0;
}

static int read_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(expected_type, data, shape, dims);
  }
}

static int write_str_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int64_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      printf("empty(");
      for (int64_t i = 1; i < rank; i++) {
        printf("[%d]", shape[i]);
      }
      printf("%s", elem_type->type_name);
      printf(")");
    } else if (rank==1) {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    } else {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    }
  }
  return 0;
}

static int write_bin_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fputs(elem_type->binname, out);
  if (shape != NULL) {
    fwrite(shape, sizeof(int64_t), rank, out);
  }

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, elem_type->size, num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type, void *data, int64_t *shape, int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary()) {
    char buf[100];
    next_token(buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(expected_type);
    size_t elem_size = expected_type->size;
    int num_elems_read = fread(dest, elem_size, 1, stdin);
    if (IS_BIG_ENDIAN) {
      flip_bytes(elem_size, (unsigned char*) dest);
    }
    return num_elems_read == 1 ? 0 : 1;
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

// End of values.h.

static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
// Start of tuning.h.

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_size)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = (char*) malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      int value = atoi(eql+1);
      if (set_size(cfg, line, value) != 0) {
        strncpy(eql+1, line, max_line_len-strlen(line)-1);
        snprintf(line, max_line_len, "Unknown name '%s' on line %d.", eql+1, lineno);
        return line;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h.

int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {"platform", required_argument, NULL,
                                            7}, {"device", required_argument,
                                                 NULL, 8},
                                           {"default-group-size",
                                            required_argument, NULL, 9},
                                           {"default-num-groups",
                                            required_argument, NULL, 10},
                                           {"default-tile-size",
                                            required_argument, NULL, 11},
                                           {"default-threshold",
                                            required_argument, NULL, 12},
                                           {"dump-opencl", required_argument,
                                            NULL, 13}, {"load-opencl",
                                                        required_argument, NULL,
                                                        14},
                                           {"dump-opencl-binary",
                                            required_argument, NULL, 15},
                                           {"load-opencl-binary",
                                            required_argument, NULL, 16},
                                           {"build-option", required_argument,
                                            NULL, 17}, {"print-sizes",
                                                        no_argument, NULL, 18},
                                           {"size", required_argument, NULL,
                                            19}, {"tuning", required_argument,
                                                  NULL, 20}, {"profile",
                                                              no_argument, NULL,
                                                              21}, {0, 0, 0,
                                                                    0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:bp:d:P", long_options,
                             NULL)) != -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                panic(1, "Cannot open %s: %s\n", optarg, strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                panic(1, "Need a positive number of runs, not %s\n", optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e') {
            if (entry_point != NULL)
                entry_point = optarg;
        }
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == 7 || ch == 'p')
            futhark_context_config_set_platform(cfg, optarg);
        if (ch == 8 || ch == 'd')
            futhark_context_config_set_device(cfg, optarg);
        if (ch == 9)
            futhark_context_config_set_default_group_size(cfg, atoi(optarg));
        if (ch == 10)
            futhark_context_config_set_default_num_groups(cfg, atoi(optarg));
        if (ch == 11)
            futhark_context_config_set_default_tile_size(cfg, atoi(optarg));
        if (ch == 12)
            futhark_context_config_set_default_threshold(cfg, atoi(optarg));
        if (ch == 13) {
            futhark_context_config_dump_program_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 14)
            futhark_context_config_load_program_from(cfg, optarg);
        if (ch == 15) {
            futhark_context_config_dump_binary_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 16)
            futhark_context_config_load_binary_from(cfg, optarg);
        if (ch == 17)
            futhark_context_config_add_build_option(cfg, optarg);
        if (ch == 18) {
            int n = futhark_get_num_sizes();
            
            for (int i = 0; i < n; i++)
                printf("%s (%s)\n", futhark_get_size_name(i),
                       futhark_get_size_class(i));
            exit(0);
        }
        if (ch == 19) {
            char *name = optarg;
            char *equals = strstr(optarg, "=");
            char *value_str = equals != NULL ? equals + 1 : optarg;
            int value = atoi(value_str);
            
            if (equals != NULL) {
                *equals = 0;
                if (futhark_context_config_set_size(cfg, name, value) != 0)
                    panic(1, "Unknown size: %s\n", name);
            } else
                panic(1, "Invalid argument for size option: %s\n", optarg);
        }
        if (ch == 20) {
            char *ret = load_tuning_file(optarg, cfg, (int (*)(void *, const
                                                               char *,
                                                               size_t)) futhark_context_config_set_size);
            
            if (ret != NULL)
                panic(1, "When loading tuning from '%s': %s\n", optarg, ret);
        }
        if (ch == 21 || ch == 'P')
            futhark_context_config_set_profiling(cfg, 1);
        if (ch == ':')
            panic(-1, "Missing argument for option %s\n", argv[optind - 1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s: %s\n", fut_progname,
                    "[-t/--write-runtime-to FILE] [-r/--runs INT] [-D/--debugging] [-L/--log] [-e/--entry-point NAME] [-b/--binary-output] [-p/--platform NAME] [-d/--device NAME] [--default-group-size INT] [--default-num-groups INT] [--default-tile-size INT] [--default-threshold INT] [--dump-opencl FILE] [--load-opencl FILE] [--dump-opencl-binary FILE] [--load-opencl-binary FILE] [--build-option OPT] [--print-sizes] [--size NAME=INT] [--tuning FILE] [-P/--profile]");
            panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
static void futrts_cli_entry_main(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_i32_1d *result_7196;
    struct futhark_i32_1d *result_7197;
    
    if (perform_warmup) {
        int r;
        
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_7196, &result_7197);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_i32_1d(ctx, result_7196) == 0);
        assert(futhark_free_i32_1d(ctx, result_7197) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_7196, &result_7197);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        if (run < num_runs - 1) {
            assert(futhark_free_i32_1d(ctx, result_7196) == 0);
            assert(futhark_free_i32_1d(ctx, result_7197) == 0);
        }
    }
    if (binary_output)
        set_binary_mode(stdout);
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_7196)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_7196, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_7196), 1);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_7197)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_7197, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_7197), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_i32_1d(ctx, result_7196) == 0);
    assert(futhark_free_i32_1d(ctx, result_7197) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="main", .fun =
                                                futrts_cli_entry_main}};
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    if (entry_point != NULL) {
        int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
        entry_point_fun *entry_point_fun = NULL;
        
        for (int i = 0; i < num_entry_points; i++) {
            if (strcmp(entry_points[i].name, entry_point) == 0) {
                entry_point_fun = entry_points[i].fun;
                break;
            }
        }
        if (entry_point_fun == NULL) {
            fprintf(stderr,
                    "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                    entry_point);
            for (int i = 0; i < num_entry_points; i++)
                fprintf(stderr, "%s\n", entry_points[i].name);
            return 1;
        }
        entry_point_fun(ctx);
        if (runtime_file != NULL)
            fclose(runtime_file);
        futhark_debugging_report(ctx);
    }
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
// Start of lock.h.

/* A very simple cross-platform implementation of locks.  Uses
   pthreads on Unix and some Windows thing there.  Futhark's
   host-level code is not multithreaded, but user code may be, so we
   need some mechanism for ensuring atomic access to API functions.
   This is that mechanism.  It is not exposed to user code at all, so
   we do not have to worry about name collisions. */

#ifdef _WIN32

typedef HANDLE lock_t;

static lock_t create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  /* Default security attributes. */
                      FALSE, /* Initially unlocked. */
                      NULL); /* Unnamed. */
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
/* Assuming POSIX */

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  /* Nothing to do for pthreads. */
  (void)lock;
}

#endif

// End of lock.h.

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_SILENCE_DEPRECATION // For macOS.
#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif
typedef cl_mem fl_mem_t;
// Start of free_list.h.

/* An entry in the free list.  May be invalid, to avoid having to
   deallocate entries as soon as they are removed.  There is also a
   tag, to help with memory reuse. */
struct free_list_entry {
  size_t size;
  fl_mem_t mem;
  const char *tag;
  unsigned char valid;
};

struct free_list {
  struct free_list_entry *entries;        // Pointer to entries.
  int capacity;                           // Number of entries.
  int used;                               // Number of valid entries.
};

static void free_list_init(struct free_list *l) {
  l->capacity = 30; // Picked arbitrarily.
  l->used = 0;
  l->entries = (struct free_list_entry*) malloc(sizeof(struct free_list_entry) * l->capacity);
  for (int i = 0; i < l->capacity; i++) {
    l->entries[i].valid = 0;
  }
}

/* Remove invalid entries from the free list. */
static void free_list_pack(struct free_list *l) {
  int p = 0;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[p] = l->entries[i];
      p++;
    }
  }
  // Now p == l->used.
  l->entries = realloc(l->entries, l->used * sizeof(struct free_list_entry));
  l->capacity = l->used;
}

static void free_list_destroy(struct free_list *l) {
  assert(l->used == 0);
  free(l->entries);
}

static int free_list_find_invalid(struct free_list *l) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (!l->entries[i].valid) {
      break;
    }
  }
  return i;
}

static void free_list_insert(struct free_list *l, size_t size, fl_mem_t mem, const char *tag) {
  int i = free_list_find_invalid(l);

  if (i == l->capacity) {
    // List is full; so we have to grow it.
    int new_capacity = l->capacity * 2 * sizeof(struct free_list_entry);
    l->entries = realloc(l->entries, new_capacity);
    for (int j = 0; j < l->capacity; j++) {
      l->entries[j+l->capacity].valid = 0;
    }
    l->capacity *= 2;
  }

  // Now 'i' points to the first invalid entry.
  l->entries[i].valid = 1;
  l->entries[i].size = size;
  l->entries[i].mem = mem;
  l->entries[i].tag = tag;

  l->used++;
}

/* Find and remove a memory block of at least the desired size and
   tag.  Returns 0 on success.  */
static int free_list_find(struct free_list *l, const char *tag, size_t *size_out, fl_mem_t *mem_out) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid && l->entries[i].tag == tag) {
      l->entries[i].valid = 0;
      *size_out = l->entries[i].size;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

/* Remove the first block in the free list.  Returns 0 if a block was
   removed, and nonzero if the free list was already empty. */
static int free_list_first(struct free_list *l, fl_mem_t *mem_out) {
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[i].valid = 0;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

// End of free_list.h.

// Start of opencl.h.

#define OPENCL_SUCCEED_FATAL(e) opencl_succeed_fatal(e, #e, __FILE__, __LINE__)
#define OPENCL_SUCCEED_NONFATAL(e) opencl_succeed_nonfatal(e, #e, __FILE__, __LINE__)
// Take care not to override an existing error.
#define OPENCL_SUCCEED_OR_RETURN(e) {             \
    char *error = OPENCL_SUCCEED_NONFATAL(e);     \
    if (error) {                                  \
      if (!ctx->error) {                          \
        ctx->error = error;                       \
        return bad;                               \
      } else {                                    \
        free(error);                              \
      }                                           \
    }                                             \
  }

// OPENCL_SUCCEED_OR_RETURN returns the value of the variable 'bad' in
// scope.  By default, it will be this one.  Create a local variable
// of some other type if needed.  This is a bit of a hack, but it
// saves effort in the code generator.
static const int bad = 1;

struct opencl_config {
  int debugging;
  int profiling;
  int logging;
  int preferred_device_num;
  const char *preferred_platform;
  const char *preferred_device;
  int ignore_blacklist;

  const char* dump_program_to;
  const char* load_program_from;
  const char* dump_binary_to;
  const char* load_binary_from;

  size_t default_group_size;
  size_t default_num_groups;
  size_t default_tile_size;
  size_t default_threshold;

  int default_group_size_changed;
  int default_tile_size_changed;

  int num_sizes;
  const char **size_names;
  const char **size_vars;
  size_t *size_values;
  const char **size_classes;
};

static void opencl_config_init(struct opencl_config *cfg,
                               int num_sizes,
                               const char *size_names[],
                               const char *size_vars[],
                               size_t *size_values,
                               const char *size_classes[]) {
  cfg->debugging = 0;
  cfg->logging = 0;
  cfg->profiling = 0;
  cfg->preferred_device_num = 0;
  cfg->preferred_platform = "";
  cfg->preferred_device = "";
  cfg->ignore_blacklist = 0;
  cfg->dump_program_to = NULL;
  cfg->load_program_from = NULL;
  cfg->dump_binary_to = NULL;
  cfg->load_binary_from = NULL;

  // The following are dummy sizes that mean the concrete defaults
  // will be set during initialisation via hardware-inspection-based
  // heuristics.
  cfg->default_group_size = 0;
  cfg->default_num_groups = 0;
  cfg->default_tile_size = 0;
  cfg->default_threshold = 0;

  cfg->default_group_size_changed = 0;
  cfg->default_tile_size_changed = 0;

  cfg->num_sizes = num_sizes;
  cfg->size_names = size_names;
  cfg->size_vars = size_vars;
  cfg->size_values = size_values;
  cfg->size_classes = size_classes;
}

// A record of something that happened.
struct profiling_record {
  cl_event *event;
  int *runs;
  int64_t *runtime;
};

struct opencl_context {
  cl_device_id device;
  cl_context ctx;
  cl_command_queue queue;

  struct opencl_config cfg;

  struct free_list free_list;

  size_t max_group_size;
  size_t max_num_groups;
  size_t max_tile_size;
  size_t max_threshold;
  size_t max_local_memory;

  size_t lockstep_width;

  struct profiling_record *profiling_records;
  int profiling_records_capacity;
  int profiling_records_used;
};

struct opencl_device_option {
  cl_platform_id platform;
  cl_device_id device;
  cl_device_type device_type;
  char *platform_name;
  char *device_name;
};

/* This function must be defined by the user.  It is invoked by
   setup_opencl() after the platform and device has been found, but
   before the program is loaded.  Its intended use is to tune
   constants based on the selected platform and device. */
static void post_opencl_setup(struct opencl_context*, struct opencl_device_option*);

static char *strclone(const char *str) {
  size_t size = strlen(str) + 1;
  char *copy = (char*) malloc(size);
  if (copy == NULL) {
    return NULL;
  }

  memcpy(copy, str, size);
  return copy;
}

// Read a file into a NUL-terminated string; returns NULL on error.
static char* slurp_file(const char *filename, size_t *size) {
  char *s;
  FILE *f = fopen(filename, "rb"); // To avoid Windows messing with linebreaks.
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  size_t src_size = ftell(f);
  fseek(f, 0, SEEK_SET);
  s = (char*) malloc(src_size + 1);
  if (fread(s, 1, src_size, f) != src_size) {
    free(s);
    s = NULL;
  } else {
    s[src_size] = '\0';
  }
  fclose(f);

  if (size) {
    *size = src_size;
  }

  return s;
}

static const char* opencl_error_string(cl_int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}

static void opencl_succeed_fatal(unsigned int ret,
                                 const char *call,
                                 const char *file,
                                 int line) {
  if (ret != CL_SUCCESS) {
    panic(-1, "%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
          file, line, call, ret, opencl_error_string(ret));
  }
}

static char* opencl_succeed_nonfatal(unsigned int ret,
                                     const char *call,
                                     const char *file,
                                     int line) {
  if (ret != CL_SUCCESS) {
    return msgprintf("%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
                     file, line, call, ret, opencl_error_string(ret));
  } else {
    return NULL;
  }
}

static void set_preferred_platform(struct opencl_config *cfg, const char *s) {
  cfg->preferred_platform = s;
  cfg->ignore_blacklist = 1;
}

static void set_preferred_device(struct opencl_config *cfg, const char *s) {
  int x = 0;
  if (*s == '#') {
    s++;
    while (isdigit(*s)) {
      x = x * 10 + (*s++)-'0';
    }
    // Skip trailing spaces.
    while (isspace(*s)) {
      s++;
    }
  }
  cfg->preferred_device = s;
  cfg->preferred_device_num = x;
  cfg->ignore_blacklist = 1;
}

static char* opencl_platform_info(cl_platform_id platform,
                                  cl_platform_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED_FATAL(clGetPlatformInfo(platform, param, 0, NULL, &req_bytes));

  info = (char*) malloc(req_bytes);

  OPENCL_SUCCEED_FATAL(clGetPlatformInfo(platform, param, req_bytes, info, NULL));

  return info;
}

static char* opencl_device_info(cl_device_id device,
                                cl_device_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device, param, 0, NULL, &req_bytes));

  info = (char*) malloc(req_bytes);

  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device, param, req_bytes, info, NULL));

  return info;
}

static void opencl_all_device_options(struct opencl_device_option **devices_out,
                                      size_t *num_devices_out) {
  size_t num_devices = 0, num_devices_added = 0;

  cl_platform_id *all_platforms;
  cl_uint *platform_num_devices;

  cl_uint num_platforms;

  // Find the number of platforms.
  OPENCL_SUCCEED_FATAL(clGetPlatformIDs(0, NULL, &num_platforms));

  // Make room for them.
  all_platforms = calloc(num_platforms, sizeof(cl_platform_id));
  platform_num_devices = calloc(num_platforms, sizeof(cl_uint));

  // Fetch all the platforms.
  OPENCL_SUCCEED_FATAL(clGetPlatformIDs(num_platforms, all_platforms, NULL));

  // Count the number of devices for each platform, as well as the
  // total number of devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    if (clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL,
                       0, NULL, &platform_num_devices[i]) == CL_SUCCESS) {
      num_devices += platform_num_devices[i];
    } else {
      platform_num_devices[i] = 0;
    }
  }

  // Make room for all the device options.
  struct opencl_device_option *devices =
    calloc(num_devices, sizeof(struct opencl_device_option));

  // Loop through the platforms, getting information about their devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    cl_platform_id platform = all_platforms[i];
    cl_uint num_platform_devices = platform_num_devices[i];

    if (num_platform_devices == 0) {
      continue;
    }

    char *platform_name = opencl_platform_info(platform, CL_PLATFORM_NAME);
    cl_device_id *platform_devices =
      calloc(num_platform_devices, sizeof(cl_device_id));

    // Fetch all the devices.
    OPENCL_SUCCEED_FATAL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                  num_platform_devices, platform_devices, NULL));

    // Loop through the devices, adding them to the devices array.
    for (cl_uint i = 0; i < num_platform_devices; i++) {
      char *device_name = opencl_device_info(platform_devices[i], CL_DEVICE_NAME);
      devices[num_devices_added].platform = platform;
      devices[num_devices_added].device = platform_devices[i];
      OPENCL_SUCCEED_FATAL(clGetDeviceInfo(platform_devices[i], CL_DEVICE_TYPE,
                                     sizeof(cl_device_type),
                                     &devices[num_devices_added].device_type,
                                     NULL));
      // We don't want the structs to share memory, so copy the platform name.
      // Each device name is already unique.
      devices[num_devices_added].platform_name = strclone(platform_name);
      devices[num_devices_added].device_name = device_name;
      num_devices_added++;
    }
    free(platform_devices);
    free(platform_name);
  }
  free(all_platforms);
  free(platform_num_devices);

  *devices_out = devices;
  *num_devices_out = num_devices;
}

// Returns 0 on success.
static int select_device_interactively(struct opencl_config *cfg) {
  struct opencl_device_option *devices;
  size_t num_devices;
  int ret = 1;

  opencl_all_device_options(&devices, &num_devices);

  printf("Choose OpenCL device:\n");
  const char *cur_platform = "";
  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strcmp(cur_platform, device.platform_name) != 0) {
      printf("Platform: %s\n", device.platform_name);
      cur_platform = device.platform_name;
    }
    printf("[%d] %s\n", (int)i, device.device_name);
  }

  int selection;
  printf("Choice: ");
  if (scanf("%d", &selection) == 1) {
    ret = 0;
    cfg->preferred_platform = "";
    cfg->preferred_device = "";
    cfg->preferred_device_num = selection;
    cfg->ignore_blacklist = 1;
  }

  // Free all the platform and device names.
  for (size_t j = 0; j < num_devices; j++) {
    free(devices[j].platform_name);
    free(devices[j].device_name);
  }
  free(devices);

  return ret;
}

static int is_blacklisted(const char *platform_name, const char *device_name,
                          const struct opencl_config *cfg) {
  if (strcmp(cfg->preferred_platform, "") != 0 ||
      strcmp(cfg->preferred_device, "") != 0) {
    return 0;
  } else if (strstr(platform_name, "Apple") != NULL &&
             strstr(device_name, "Intel(R) Core(TM)") != NULL) {
    return 1;
  } else {
    return 0;
  }
}

static struct opencl_device_option get_preferred_device(const struct opencl_config *cfg) {
  struct opencl_device_option *devices;
  size_t num_devices;

  opencl_all_device_options(&devices, &num_devices);

  int num_device_matches = 0;

  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strstr(device.platform_name, cfg->preferred_platform) != NULL &&
        strstr(device.device_name, cfg->preferred_device) != NULL &&
        (cfg->ignore_blacklist ||
         !is_blacklisted(device.platform_name, device.device_name, cfg)) &&
        num_device_matches++ == cfg->preferred_device_num) {
      // Free all the platform and device names, except the ones we have chosen.
      for (size_t j = 0; j < num_devices; j++) {
        if (j != i) {
          free(devices[j].platform_name);
          free(devices[j].device_name);
        }
      }
      free(devices);
      return device;
    }
  }

  panic(1, "Could not find acceptable OpenCL device.\n");
  exit(1); // Never reached
}

static void describe_device_option(struct opencl_device_option device) {
  fprintf(stderr, "Using platform: %s\n", device.platform_name);
  fprintf(stderr, "Using device: %s\n", device.device_name);
}

static cl_build_status build_opencl_program(cl_program program, cl_device_id device, const char* options) {
  cl_int clBuildProgram_error = clBuildProgram(program, 1, &device, options, NULL, NULL);

  // Avoid termination due to CL_BUILD_PROGRAM_FAILURE
  if (clBuildProgram_error != CL_SUCCESS &&
      clBuildProgram_error != CL_BUILD_PROGRAM_FAILURE) {
    OPENCL_SUCCEED_FATAL(clBuildProgram_error);
  }

  cl_build_status build_status;
  OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program,
                                             device,
                                             CL_PROGRAM_BUILD_STATUS,
                                             sizeof(cl_build_status),
                                             &build_status,
                                             NULL));

  if (build_status != CL_SUCCESS) {
    char *build_log;
    size_t ret_val_size;
    OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size));

    build_log = (char*) malloc(ret_val_size+1);
    OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL));

    // The spec technically does not say whether the build log is zero-terminated, so let's be careful.
    build_log[ret_val_size] = '\0';

    fprintf(stderr, "Build log:\n%s\n", build_log);

    free(build_log);
  }

  return build_status;
}

/* Fields in a bitmask indicating which types we must be sure are
   available. */
enum opencl_required_type { OPENCL_F64 = 1 };

// We take as input several strings representing the program, because
// C does not guarantee that the compiler supports particularly large
// literals.  Notably, Visual C has a limit of 2048 characters.  The
// array must be NULL-terminated.
static cl_program setup_opencl_with_command_queue(struct opencl_context *ctx,
                                                  cl_command_queue queue,
                                                  const char *srcs[],
                                                  int required_types,
                                                  const char *extra_build_opts[]) {
  int error;

  ctx->queue = queue;

  OPENCL_SUCCEED_FATAL(clGetCommandQueueInfo(ctx->queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx->ctx, NULL));

  // Fill out the device info.  This is redundant work if we are
  // called from setup_opencl() (which is the common case), but I
  // doubt it matters much.
  struct opencl_device_option device_option;
  OPENCL_SUCCEED_FATAL(clGetCommandQueueInfo(ctx->queue, CL_QUEUE_DEVICE,
                                       sizeof(cl_device_id),
                                       &device_option.device,
                                       NULL));
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_PLATFORM,
                                 sizeof(cl_platform_id),
                                 &device_option.platform,
                                 NULL));
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_TYPE,
                                 sizeof(cl_device_type),
                                 &device_option.device_type,
                                 NULL));
  device_option.platform_name = opencl_platform_info(device_option.platform, CL_PLATFORM_NAME);
  device_option.device_name = opencl_device_info(device_option.device, CL_DEVICE_NAME);

  ctx->device = device_option.device;

  if (required_types & OPENCL_F64) {
    cl_uint supported;
    OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                                   sizeof(cl_uint), &supported, NULL));
    if (!supported) {
      panic(1, "Program uses double-precision floats, but this is not supported on the chosen device: %s\n",
            device_option.device_name);
    }
  }

  size_t max_group_size;
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(size_t), &max_group_size, NULL));

  size_t max_tile_size = sqrt(max_group_size);

  cl_ulong max_local_memory;
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_LOCAL_MEM_SIZE,
                                       sizeof(size_t), &max_local_memory, NULL));

  // Make sure this function is defined.
  post_opencl_setup(ctx, &device_option);

  if (max_group_size < ctx->cfg.default_group_size) {
    if (ctx->cfg.default_group_size_changed) {
      fprintf(stderr, "Note: Device limits default group size to %zu (down from %zu).\n",
              max_group_size, ctx->cfg.default_group_size);
    }
    ctx->cfg.default_group_size = max_group_size;
  }

  if (max_tile_size < ctx->cfg.default_tile_size) {
    if (ctx->cfg.default_tile_size_changed) {
      fprintf(stderr, "Note: Device limits default tile size to %zu (down from %zu).\n",
              max_tile_size, ctx->cfg.default_tile_size);
    }
    ctx->cfg.default_tile_size = max_tile_size;
  }

  ctx->max_group_size = max_group_size;
  ctx->max_tile_size = max_tile_size; // No limit.
  ctx->max_threshold = ctx->max_num_groups = 0; // No limit.
  ctx->max_local_memory = max_local_memory;

  // Now we go through all the sizes, clamp them to the valid range,
  // or set them to the default.
  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    const char *size_class = ctx->cfg.size_classes[i];
    size_t *size_value = &ctx->cfg.size_values[i];
    const char* size_name = ctx->cfg.size_names[i];
    size_t max_value, default_value;
    if (strstr(size_class, "group_size") == size_class) {
      max_value = max_group_size;
      default_value = ctx->cfg.default_group_size;
    } else if (strstr(size_class, "num_groups") == size_class) {
      max_value = max_group_size; // Futhark assumes this constraint.
      default_value = ctx->cfg.default_num_groups;
    } else if (strstr(size_class, "tile_size") == size_class) {
      max_value = sqrt(max_group_size);
      default_value = ctx->cfg.default_tile_size;
    } else if (strstr(size_class, "threshold") == size_class) {
      max_value = 0; // No limit.
      default_value = ctx->cfg.default_threshold;
    } else {
      panic(1, "Unknown size class for size '%s': %s\n", size_name, size_class);
    }
    if (*size_value == 0) {
      *size_value = default_value;
    } else if (max_value > 0 && *size_value > max_value) {
      fprintf(stderr, "Note: Device limits %s to %d (down from %d)\n",
              size_name, (int)max_value, (int)*size_value);
      *size_value = max_value;
    }
  }

  if (ctx->lockstep_width == 0) {
    ctx->lockstep_width = 1;
  }

  if (ctx->cfg.logging) {
    fprintf(stderr, "Lockstep width: %d\n", (int)ctx->lockstep_width);
    fprintf(stderr, "Default group size: %d\n", (int)ctx->cfg.default_group_size);
    fprintf(stderr, "Default number of groups: %d\n", (int)ctx->cfg.default_num_groups);
  }

  char *fut_opencl_src = NULL;
  size_t src_size = 0;

  // Maybe we have to read OpenCL source from somewhere else (used for debugging).
  if (ctx->cfg.load_program_from != NULL) {
    fut_opencl_src = slurp_file(ctx->cfg.load_program_from, NULL);
    assert(fut_opencl_src != NULL);
  } else {
    // Build the OpenCL program.  First we have to concatenate all the fragments.
    for (const char **src = srcs; src && *src; src++) {
      src_size += strlen(*src);
    }

    fut_opencl_src = (char*) malloc(src_size + 1);

    size_t n, i;
    for (i = 0, n = 0; srcs && srcs[i]; i++) {
      strncpy(fut_opencl_src+n, srcs[i], src_size-n);
      n += strlen(srcs[i]);
    }
    fut_opencl_src[src_size] = 0;

  }

  cl_program prog;
  error = CL_SUCCESS;
  const char* src_ptr[] = {fut_opencl_src};

  if (ctx->cfg.dump_program_to != NULL) {
    FILE *f = fopen(ctx->cfg.dump_program_to, "w");
    assert(f != NULL);
    fputs(fut_opencl_src, f);
    fclose(f);
  }

  if (ctx->cfg.load_binary_from == NULL) {
    prog = clCreateProgramWithSource(ctx->ctx, 1, src_ptr, &src_size, &error);
    OPENCL_SUCCEED_FATAL(error);

    int compile_opts_size = 1024;

    for (int i = 0; i < ctx->cfg.num_sizes; i++) {
      compile_opts_size += strlen(ctx->cfg.size_names[i]) + 20;
    }

    for (int i = 0; extra_build_opts[i] != NULL; i++) {
      compile_opts_size += strlen(extra_build_opts[i] + 1);
    }

    char *compile_opts = (char*) malloc(compile_opts_size);

    int w = snprintf(compile_opts, compile_opts_size,
                     "-DLOCKSTEP_WIDTH=%d ",
                     (int)ctx->lockstep_width);

    for (int i = 0; i < ctx->cfg.num_sizes; i++) {
      w += snprintf(compile_opts+w, compile_opts_size-w,
                    "-D%s=%d ",
                    ctx->cfg.size_vars[i],
                    (int)ctx->cfg.size_values[i]);
    }

    for (int i = 0; extra_build_opts[i] != NULL; i++) {
      w += snprintf(compile_opts+w, compile_opts_size-w,
                    "%s ", extra_build_opts[i]);
    }

    OPENCL_SUCCEED_FATAL(build_opencl_program(prog, device_option.device, compile_opts));

    free(compile_opts);
  } else {
    size_t binary_size;
    unsigned char *fut_opencl_bin =
      (unsigned char*) slurp_file(ctx->cfg.load_binary_from, &binary_size);
    assert(fut_opencl_src != NULL);
    const unsigned char *binaries[1] = { fut_opencl_bin };
    cl_int status = 0;

    prog = clCreateProgramWithBinary(ctx->ctx, 1, &device_option.device,
                                     &binary_size, binaries,
                                     &status, &error);

    OPENCL_SUCCEED_FATAL(status);
    OPENCL_SUCCEED_FATAL(error);
  }

  free(fut_opencl_src);

  if (ctx->cfg.dump_binary_to != NULL) {
    size_t binary_size;
    OPENCL_SUCCEED_FATAL(clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES,
                                          sizeof(size_t), &binary_size, NULL));
    unsigned char *binary = (unsigned char*) malloc(binary_size);
    unsigned char *binaries[1] = { binary };
    OPENCL_SUCCEED_FATAL(clGetProgramInfo(prog, CL_PROGRAM_BINARIES,
                                          sizeof(unsigned char*), binaries, NULL));

    FILE *f = fopen(ctx->cfg.dump_binary_to, "w");
    assert(f != NULL);
    fwrite(binary, sizeof(char), binary_size, f);
    fclose(f);
  }

  return prog;
}

static cl_program setup_opencl(struct opencl_context *ctx,
                               const char *srcs[],
                               int required_types,
                               const char *extra_build_opts[]) {

  ctx->lockstep_width = 0; // Real value set later.

  free_list_init(&ctx->free_list);

  struct opencl_device_option device_option = get_preferred_device(&ctx->cfg);

  if (ctx->cfg.logging) {
    describe_device_option(device_option);
  }

  // Note that NVIDIA's OpenCL requires the platform property
  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)device_option.platform,
    0
  };

  cl_int clCreateContext_error;
  ctx->ctx = clCreateContext(properties, 1, &device_option.device, NULL, NULL, &clCreateContext_error);
  OPENCL_SUCCEED_FATAL(clCreateContext_error);

  cl_int clCreateCommandQueue_error;
  cl_command_queue queue =
    clCreateCommandQueue(ctx->ctx,
                         device_option.device,
                         ctx->cfg.profiling ? CL_QUEUE_PROFILING_ENABLE : 0,
                         &clCreateCommandQueue_error);
  OPENCL_SUCCEED_FATAL(clCreateCommandQueue_error);

  return setup_opencl_with_command_queue(ctx, queue, srcs, required_types, extra_build_opts);
}

// Count up the runtime all the profiling_records that occured during execution.
// Also clears the buffer of profiling_records.
static cl_int opencl_tally_profiling_records(struct opencl_context *ctx) {
  cl_int err;
  for (int i = 0; i < ctx->profiling_records_used; i++) {
    struct profiling_record record = ctx->profiling_records[i];

    cl_ulong start_t, end_t;

    if ((err = clGetEventProfilingInfo(*record.event,
                                       CL_PROFILING_COMMAND_START,
                                       sizeof(start_t),
                                       &start_t,
                                       NULL)) != CL_SUCCESS) {
      return err;
    }

    if ((err = clGetEventProfilingInfo(*record.event,
                                       CL_PROFILING_COMMAND_END,
                                       sizeof(end_t),
                                       &end_t,
                                       NULL)) != CL_SUCCESS) {
      return err;
    }

    // OpenCL provides nanosecond resolution, but we want
    // microseconds.
    *record.runs += 1;
    *record.runtime += (end_t - start_t)/1000;

    if ((err = clReleaseEvent(*record.event)) != CL_SUCCESS) {
      return err;
    }
    free(record.event);
  }

  ctx->profiling_records_used = 0;

  return CL_SUCCESS;
}

// If profiling, produce an event associated with a profiling record.
static cl_event* opencl_get_event(struct opencl_context *ctx, int *runs, int64_t *runtime) {
    if (ctx->profiling_records_used == ctx->profiling_records_capacity) {
      ctx->profiling_records_capacity *= 2;
      ctx->profiling_records =
        realloc(ctx->profiling_records,
                ctx->profiling_records_capacity *
                sizeof(struct profiling_record));
    }
    cl_event *event = malloc(sizeof(cl_event));
    ctx->profiling_records[ctx->profiling_records_used].event = event;
    ctx->profiling_records[ctx->profiling_records_used].runs = runs;
    ctx->profiling_records[ctx->profiling_records_used].runtime = runtime;
    ctx->profiling_records_used++;
    return event;
}

// Allocate memory from driver. The problem is that OpenCL may perform
// lazy allocation, so we cannot know whether an allocation succeeded
// until the first time we try to use it.  Hence we immediately
// perform a write to see if the allocation succeeded.  This is slow,
// but the assumption is that this operation will be rare (most things
// will go through the free list).
static int opencl_alloc_actual(struct opencl_context *ctx, size_t size, cl_mem *mem_out) {
  int error;
  *mem_out = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, size, NULL, &error);

  if (error != CL_SUCCESS) {
    return error;
  }

  int x = 2;
  error = clEnqueueWriteBuffer(ctx->queue, *mem_out, 1, 0, sizeof(x), &x, 0, NULL, NULL);

  // No need to wait for completion here. clWaitForEvents() cannot
  // return mem object allocation failures. This implies that the
  // buffer is faulted onto the device on enqueue. (Observation by
  // Andreas Kloeckner.)

  return error;
}

static int opencl_alloc(struct opencl_context *ctx, size_t min_size, const char *tag, cl_mem *mem_out) {
  if (min_size < sizeof(int)) {
    min_size = sizeof(int);
  }

  size_t size;

  if (free_list_find(&ctx->free_list, tag, &size, mem_out) == 0) {
    // Successfully found a free block.  Is it big enough?
    //
    // FIXME: we might also want to check whether the block is *too
    // big*, to avoid internal fragmentation.  However, this can
    // sharply impact performance on programs where arrays change size
    // frequently.  Fortunately, such allocations are usually fairly
    // short-lived, as they are necessarily within a loop, so the risk
    // of internal fragmentation resulting in an OOM situation is
    // limited.  However, it would be preferable if we could go back
    // and *shrink* oversize allocations when we encounter an OOM
    // condition.  That is technically feasible, since we do not
    // expose OpenCL pointer values directly to the application, but
    // instead rely on a level of indirection.
    if (size >= min_size) {
      return CL_SUCCESS;
    } else {
      // Not just right - free it.
      int error = clReleaseMemObject(*mem_out);
      if (error != CL_SUCCESS) {
        return error;
      }
    }
  }

  // We have to allocate a new block from the driver.  If the
  // allocation does not succeed, then we might be in an out-of-memory
  // situation.  We now start freeing things from the free list until
  // we think we have freed enough that the allocation will succeed.
  // Since we don't know how far the allocation is from fitting, we
  // have to check after every deallocation.  This might be pretty
  // expensive.  Let's hope that this case is hit rarely.

  int error = opencl_alloc_actual(ctx, min_size, mem_out);

  while (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
    if (ctx->cfg.debugging) {
      fprintf(stderr, "Out of OpenCL memory: releasing entry from the free list...\n");
    }
    cl_mem mem;
    if (free_list_first(&ctx->free_list, &mem) == 0) {
      error = clReleaseMemObject(mem);
      if (error != CL_SUCCESS) {
        return error;
      }
    } else {
      break;
    }
    error = opencl_alloc_actual(ctx, min_size, mem_out);
  }

  return error;
}

static int opencl_free(struct opencl_context *ctx, cl_mem mem, const char *tag) {
  size_t size;
  cl_mem existing_mem;

  // If there is already a block with this tag, then remove it.
  if (free_list_find(&ctx->free_list, tag, &size, &existing_mem) == 0) {
    int error = clReleaseMemObject(existing_mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  int error = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size_t), &size, NULL);

  if (error == CL_SUCCESS) {
    free_list_insert(&ctx->free_list, size, mem, tag);
  }

  return error;
}

static int opencl_free_all(struct opencl_context *ctx) {
  cl_mem mem;
  free_list_pack(&ctx->free_list);
  while (free_list_first(&ctx->free_list, &mem) == 0) {
    int error = clReleaseMemObject(mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  return CL_SUCCESS;
}

// End of opencl.h.

static const char *opencl_program[] =
                  {"#ifdef cl_clang_storage_class_specifiers\n#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable\n#endif\n#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n__kernel void dummy_kernel(__global unsigned char *dummy, int n)\n{\n    const int thread_gid = get_global_id(0);\n    \n    if (thread_gid >= n)\n        return;\n}\ntypedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long int64_t;\ntypedef uchar uint8_t;\ntypedef ushort uint16_t;\ntypedef uint uint32_t;\ntypedef ulong uint64_t;\n#ifdef cl_nv_pragma_unroll\nstatic inline void mem_fence_global()\n{\n    asm(\"membar.gl;\");\n}\n#else\nstatic inline void mem_fence_global()\n{\n    mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n}\n#endif\nstatic inline void mem_fence_local()\n{\n    mem_fence(CLK_LOCAL_MEM_FENCE);\n}\nstatic inline int8_t add8(int8_t x, int8_t y)\n{\n    return x + y;\n}\nstatic inline int16_t add16(int16_t x, int16_t y)\n{\n    return x + y;\n}\nstatic inline int32_t add32(int32_t x, int32_t y)\n{\n    return x + y;\n}\nstatic inline int64_t add64(int64_t x, int64_t y)\n{\n    return x + y;\n}\nstatic inline int8_t sub8(int8_t x, int8_t y)\n{\n    return x - y;\n}\nstatic inline int16_t sub16(int16_t x, int16_t y)\n{\n    return x - y;\n}\nstatic inline int32_t sub32(int32_t x, int32_t y)\n{\n    return x - y;\n}\nstatic inline int64_t sub64(int64_t x, int64_t y)\n{\n    return x - y;\n}\nstatic inline int8_t mul8(int8_t x, int8_t y)\n{\n    return x * y;\n}\nstatic inline int16_t mul16(int16_t x, int16_t y)\n{\n    return x * y;\n}\nstatic inline int32_t mul32(int32_t x, int32_t y)\n{\n    return x * y;\n}\nstatic inline int64_t mul64(int64_t x, int64_t y)\n{\n    return x * y;\n}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    return x / y;\n}\nstatic inline uint64_t udiv64(uint64_t x, uint64_t y)\n{\n    return x / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint",
                   "8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstatic inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\nstatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    ret",
                   "urn x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t smin8(int8_t x, int8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int16_t smin16(int16_t x, int16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int32_t smin32(int32_t x, int32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int64_t smin64(int64_t x, int64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint8_t umin8(uint8_t x, uint8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint16_t umin16(uint16_t x, uint16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint32_t umin32(uint32_t x, uint32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint64_t umin64(uint64_t x, uint64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int8_t smax8(int8_t x, int8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int16_t smax16(int16_t x, int16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int32_t smax32(int32_t x, int32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int64_t smax64(int64_t x, int64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t umax8(uint8_t x, uint8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint16_t umax16(uint16_t x, uint16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint32_t umax32(uint32_t x, uint32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint64_t umax64(uint64_t x, uint64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr32(uint32_t x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic",
                   " inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstatic inline int32_t ashr32(int32_t x, int32_t y)\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int64_t y)\n{\n    return x >> y;\n}\nstatic inline uint8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline char ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline char ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline char ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline char ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline char ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline char ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline char ule32(uint32_t x, uint32_t y)\n{\n    return x <= y;\n}\nstatic inline char ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline char slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline char slt16(int16_t x, int16_t y)\n{\n    return x < y;\n}\nstatic inline char slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline char slt64(int64_t x, int64_t y)\n{\n    return ",
                   "x < y;\n}\nstatic inline char sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline char sle16(int16_t x, int16_t y)\n{\n    return x <= y;\n}\nstatic inline char sle32(int32_t x, int32_t y)\n{\n    return x <= y;\n}\nstatic inline char sle64(int64_t x, int64_t y)\n{\n    return x <= y;\n}\nstatic inline int8_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline bool itob_i8_bool(int8_t x)\n{\n    return x;\n}\nstatic inline bool itob_i16_bool(int16_t x)\n{\n    return x;\n}\nstatic inline bool itob_i32_bool(int32_t x)\n{\n    return x;\n}\nstatic inline bool itob_i64_bool(int64_t x)\n{\n    return x;\n}\nstatic inline int8_t btoi_bool_i8(bool x)\n{\n    return x;\n}\nstatic inline int16_t btoi_bool_i16(bool x)\n{\n    return x;\n}\nstatic inline int32_t btoi_bool_i32(bool x)\n{\n    return x;\n}\nstatic inline int64_t btoi_bool_i64(bool x)\n{\n    return x;\n}\n#define sext_i8_i8(x) ((int8_t) (int8_t) x)\n#define sext_i8_i16(x) ((int16_t) (int8_t) x)\n#define sext_i8_i32(x) ((int32_t) (int8_t) x)\n#define sext_i8_i64(x) ((int64_t) (int8_t) x)\n#define sext_i16_i8(x) ((int8_t) (int16_t) x)\n#define sext_i16_i16(x) ((int16_t) (int16_t) x)\n#define sext_i16_i32(x) ((int32_t) (int16_t) x)\n#define sext_i16_i64(x) ((int64_t) (int16_t) x)\n#define",
                   " sext_i32_i8(x) ((int8_t) (int32_t) x)\n#define sext_i32_i16(x) ((int16_t) (int32_t) x)\n#define sext_i32_i32(x) ((int32_t) (int32_t) x)\n#define sext_i32_i64(x) ((int64_t) (int32_t) x)\n#define sext_i64_i8(x) ((int8_t) (int64_t) x)\n#define sext_i64_i16(x) ((int16_t) (int64_t) x)\n#define sext_i64_i32(x) ((int32_t) (int64_t) x)\n#define sext_i64_i64(x) ((int64_t) (int64_t) x)\n#define zext_i8_i8(x) ((uint8_t) (uint8_t) x)\n#define zext_i8_i16(x) ((uint16_t) (uint8_t) x)\n#define zext_i8_i32(x) ((uint32_t) (uint8_t) x)\n#define zext_i8_i64(x) ((uint64_t) (uint8_t) x)\n#define zext_i16_i8(x) ((uint8_t) (uint16_t) x)\n#define zext_i16_i16(x) ((uint16_t) (uint16_t) x)\n#define zext_i16_i32(x) ((uint32_t) (uint16_t) x)\n#define zext_i16_i64(x) ((uint64_t) (uint16_t) x)\n#define zext_i32_i8(x) ((uint8_t) (uint32_t) x)\n#define zext_i32_i16(x) ((uint16_t) (uint32_t) x)\n#define zext_i32_i32(x) ((uint32_t) (uint32_t) x)\n#define zext_i32_i64(x) ((uint64_t) (uint32_t) x)\n#define zext_i64_i8(x) ((uint8_t) (uint64_t) x)\n#define zext_i64_i16(x) ((uint16_t) (uint64_t) x)\n#define zext_i64_i32(x) ((uint32_t) (uint64_t) x)\n#define zext_i64_i64(x) ((uint64_t) (uint64_t) x)\n#ifdef __OPENCL_VERSION__\nint32_t futrts_popc8(int8_t x)\n{\n    return popcount(x);\n}\nint32_t futrts_popc16(int16_t x)\n{\n    return popcount(x);\n}\nint32_t futrts_popc32(int32_t x)\n{\n    return popcount(x);\n}\nint32_t futrts_popc64(int64_t x)\n{\n    return popcount(x);\n}\n#elif __CUDA_ARCH__\nint32_t futrts_popc8(int8_t x)\n{\n    return __popc(zext_i8_i32(x));\n}\nint32_t futrts_popc16(int16_t x)\n{\n    return __popc(zext_i16_i32(x));\n}\nint32_t futrts_popc32(int32_t x)\n{\n    return __popc(x);\n}\nint32_t futrts_popc64(int64_t x)\n{\n    return __popcll(x);\n}\n#else\nint32_t futrts_popc8(int8_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nint32_t futrts_popc16(int16_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nint32_t futrts_popc32(int32_t x)\n{\n    int c = 0;\n    \n    for (; x;",
                   " ++c)\n        x &= x - 1;\n    return c;\n}\nint32_t futrts_popc64(int64_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\n#endif\n#ifdef __OPENCL_VERSION__\nint32_t futrts_clzz8(int8_t x)\n{\n    return clz(x);\n}\nint32_t futrts_clzz16(int16_t x)\n{\n    return clz(x);\n}\nint32_t futrts_clzz32(int32_t x)\n{\n    return clz(x);\n}\nint32_t futrts_clzz64(int64_t x)\n{\n    return clz(x);\n}\n#elif __CUDA_ARCH__\nint32_t futrts_clzz8(int8_t x)\n{\n    return __clz(zext_i8_i32(x)) - 24;\n}\nint32_t futrts_clzz16(int16_t x)\n{\n    return __clz(zext_i16_i32(x)) - 16;\n}\nint32_t futrts_clzz32(int32_t x)\n{\n    return __clz(x);\n}\nint32_t futrts_clzz64(int64_t x)\n{\n    return __clzll(x);\n}\n#else\nint32_t futrts_clzz8(int8_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nint32_t futrts_clzz16(int16_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nint32_t futrts_clzz32(int32_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nint32_t futrts_clzz64(int64_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\n#endif\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fmin32(float x, float y)\n{\n    return fmin(x, y);\n}\nstatic inline float fmax32(float x, float y)\n{\n    return fmax(x, y);\n}\nstatic inline float fpow32(float ",
                   "x, float y)\n{\n    return pow(x, y);\n}\nstatic inline char cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline char cmple32(float x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i32_f32(int32_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    return x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2",
                   "(x, y);\n}\nstatic inline float futrts_gamma32(float x)\n{\n    return tgamma(x);\n}\nstatic inline float futrts_lgamma32(float x)\n{\n    return lgamma(x);\n}\nstatic inline char futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline char futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#ifdef __OPENCL_VERSION__\nstatic inline float fmod32(float x, float y)\n{\n    return fmod(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floor(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceil(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return mix(v0, v1, t);\n}\n#else\nstatic inline float fmod32(float x, float y)\n{\n    return fmodf(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rintf(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floorf(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceilf(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return v0 + (v1 - v0) * t;\n}\n#endif\n__kernel void replicate_7035(__global unsigned char *mem_7031,\n                             int32_t num_elems_7032, int32_t val_7033)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_7035;\n    int32_t replicate_ltid_7036;\n    int32_t replicate_gid_7037;\n    \n    replicate_gtid_7035 = get_global_id(0);\n    replicate_ltid_7036 = get_local_id(0);\n    replicate_gid_7037 = get_group_id(0);\n    if (slt32(replicate_gtid_7035, num_elems_7032)) {\n        ((__global int32_t *) mem_7031)[replicate_gtid_7035] = val_7033;\n    }\n}\n__kernel void scan_stage1_6792(__local volati",
                   "le\n                               int64_t *scan_arr_mem_6979_backing_aligned_0,\n                               __local volatile\n                               int64_t *scan_arr_mem_6981_backing_aligned_1,\n                               __global unsigned char *mem_6914, __global\n                               unsigned char *mem_6926, __global\n                               unsigned char *mem_6930, __global\n                               unsigned char *mem_6933,\n                               int32_t num_threads_6967)\n{\n    const int32_t segscan_group_sizze_6787 = mainzisegscan_group_sizze_6786;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_6979_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_6979_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_6981_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_6981_backing_aligned_1;\n    int32_t global_tid_6974;\n    int32_t local_tid_6975;\n    int32_t group_sizze_6978;\n    int32_t wave_sizze_6977;\n    int32_t group_tid_6976;\n    \n    global_tid_6974 = get_global_id(0);\n    local_tid_6975 = get_local_id(0);\n    group_sizze_6978 = get_local_size(0);\n    wave_sizze_6977 = LOCKSTEP_WIDTH;\n    group_tid_6976 = get_group_id(0);\n    \n    int32_t phys_tid_6792 = global_tid_6974;\n    __local char *scan_arr_mem_6979;\n    \n    scan_arr_mem_6979 = (__local char *) scan_arr_mem_6979_backing_0;\n    \n    __local char *scan_arr_mem_6981;\n    \n    scan_arr_mem_6981 = (__local char *) scan_arr_mem_6981_backing_1;\n    \n    int32_t x_6714;\n    int32_t x_6715;\n    int32_t x_6716;\n    int32_t x_6717;\n    \n    x_6714 = 0;\n    x_6715 = 0;\n    for (int32_t j_6983 = 0; j_6983 < squot32(3 + num_threads_6967 - 1,\n                                              num_threads_6967); j_6983++) {\n        int32_t chunk_offset_6984 = segscan_group_",
                   "sizze_6787 * j_6983 +\n                group_tid_6976 * (segscan_group_sizze_6787 * squot32(3 +\n                                                                     num_threads_6967 -\n                                                                     1,\n                                                                     num_threads_6967));\n        int32_t flat_idx_6985 = chunk_offset_6984 + local_tid_6975;\n        int32_t gtid_6791 = flat_idx_6985;\n        \n        // threads in bounds read input; others get neutral element\n        {\n            if (slt32(gtid_6791, 3)) {\n                bool cond_6721 = slt32(0, gtid_6791);\n                int32_t res_6722;\n                \n                if (cond_6721) {\n                    int32_t i_6723 = gtid_6791 - 1;\n                    int32_t res_6724 = ((__global int32_t *) mem_6914)[i_6723];\n                    \n                    res_6722 = res_6724;\n                } else {\n                    res_6722 = 0;\n                }\n                \n                bool cond_6725 = gtid_6791 == 0;\n                int32_t res_6726;\n                \n                if (cond_6725) {\n                    res_6726 = 0;\n                } else {\n                    int32_t i_6727 = gtid_6791 - 1;\n                    int32_t res_6728 = ((__global int32_t *) mem_6926)[i_6727];\n                    \n                    res_6726 = res_6728;\n                }\n                // write to-scan values to parameters\n                {\n                    x_6716 = res_6722;\n                    x_6717 = res_6726;\n                }\n                // write mapped values results to global memory\n                { }\n            } else {\n                x_6716 = 0;\n                x_6717 = 0;\n            }\n        }\n        // combine with carry and write to local memory\n        {\n            int32_t res_6718 = x_6714 + x_6716;\n            int32_t res_6719 = x_6715 + x_6717;\n            \n            ((__local int32_t *) scan_arr_mem_6979)[local_tid",
                   "_6975] = res_6718;\n            ((__local int32_t *) scan_arr_mem_6981)[local_tid_6975] = res_6719;\n        }\n        \n        int32_t x_6968;\n        int32_t x_6969;\n        int32_t x_6970;\n        int32_t x_6971;\n        int32_t x_6986;\n        int32_t x_6987;\n        int32_t x_6988;\n        int32_t x_6989;\n        int32_t skip_threads_6992;\n        \n        if (slt32(local_tid_6975, segscan_group_sizze_6787)) {\n            x_6970 = ((volatile __local\n                       int32_t *) scan_arr_mem_6979)[local_tid_6975];\n            x_6971 = ((volatile __local\n                       int32_t *) scan_arr_mem_6981)[local_tid_6975];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_6992 = 1;\n            while (slt32(skip_threads_6992, 32)) {\n                if (sle32(skip_threads_6992, local_tid_6975 -\n                          squot32(local_tid_6975, 32) * 32) &&\n                    slt32(local_tid_6975, segscan_group_sizze_6787)) {\n                    // read operands\n                    {\n                        x_6968 = ((volatile __local\n                                   int32_t *) scan_arr_mem_6979)[local_tid_6975 -\n                                                                 skip_threads_6992];\n                        x_6969 = ((volatile __local\n                                   int32_t *) scan_arr_mem_6981)[local_tid_6975 -\n                                                                 skip_threads_6992];\n                    }\n                    // perform operation\n                    {\n                        int32_t res_6972 = x_6968 + x_6970;\n                        int32_t res_6973 = x_6969 + x_6971;\n                        \n                        x_6970 = res_6972;\n                        x_6971 = res_6973;\n                    }\n                }\n                if (sle32(wave_sizze_6977, skip_threads_6992)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if ",
                   "(sle32(skip_threads_6992, local_tid_6975 -\n                          squot32(local_tid_6975, 32) * 32) &&\n                    slt32(local_tid_6975, segscan_group_sizze_6787)) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_6979)[local_tid_6975] =\n                            x_6970;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_6981)[local_tid_6975] =\n                            x_6971;\n                    }\n                }\n                if (sle32(wave_sizze_6977, skip_threads_6992)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_6992 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // last thread of block 'i' writes its result to offset 'i'\n        {\n            if ((local_tid_6975 - squot32(local_tid_6975, 32) * 32) == 31 &&\n                slt32(local_tid_6975, segscan_group_sizze_6787)) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_6979)[squot32(local_tid_6975, 32)] =\n                    x_6970;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_6981)[squot32(local_tid_6975, 32)] =\n                    x_6971;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n        {\n            int32_t skip_threads_6993;\n            \n            if (squot32(local_tid_6975, 32) == 0 && slt32(local_tid_6975,\n                                                          segscan_group_sizze_6787)) {\n                x_6988 = ((volatile __local\n                           int32_t *) scan_arr_mem_6979)[local_tid_6975];\n                x_6989 = ((volatile __local\n                           int32_t *) scan_arr_mem_6981)[local_tid_6975];\n            }\n            // in-block scan (hopefully no barriers ne",
                   "eded)\n            {\n                skip_threads_6993 = 1;\n                while (slt32(skip_threads_6993, 32)) {\n                    if (sle32(skip_threads_6993, local_tid_6975 -\n                              squot32(local_tid_6975, 32) * 32) &&\n                        (squot32(local_tid_6975, 32) == 0 &&\n                         slt32(local_tid_6975, segscan_group_sizze_6787))) {\n                        // read operands\n                        {\n                            x_6986 = ((volatile __local\n                                       int32_t *) scan_arr_mem_6979)[local_tid_6975 -\n                                                                     skip_threads_6993];\n                            x_6987 = ((volatile __local\n                                       int32_t *) scan_arr_mem_6981)[local_tid_6975 -\n                                                                     skip_threads_6993];\n                        }\n                        // perform operation\n                        {\n                            int32_t res_6990 = x_6986 + x_6988;\n                            int32_t res_6991 = x_6987 + x_6989;\n                            \n                            x_6988 = res_6990;\n                            x_6989 = res_6991;\n                        }\n                    }\n                    if (sle32(wave_sizze_6977, skip_threads_6993)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_6993, local_tid_6975 -\n                              squot32(local_tid_6975, 32) * 32) &&\n                        (squot32(local_tid_6975, 32) == 0 &&\n                         slt32(local_tid_6975, segscan_group_sizze_6787))) {\n                        // write result\n                        {\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_6979)[local_tid_6975] =\n                                x_6988;\n                            ((volatile __lo",
                   "cal\n                              int32_t *) scan_arr_mem_6981)[local_tid_6975] =\n                                x_6989;\n                        }\n                    }\n                    if (sle32(wave_sizze_6977, skip_threads_6993)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_6993 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for every block except the first\n        {\n            if (!(squot32(local_tid_6975, 32) == 0 || !slt32(local_tid_6975,\n                                                             segscan_group_sizze_6787))) {\n                // read operands\n                {\n                    x_6968 = ((volatile __local\n                               int32_t *) scan_arr_mem_6979)[squot32(local_tid_6975,\n                                                                     32) - 1];\n                    x_6969 = ((volatile __local\n                               int32_t *) scan_arr_mem_6981)[squot32(local_tid_6975,\n                                                                     32) - 1];\n                }\n                // perform operation\n                {\n                    int32_t res_6972 = x_6968 + x_6970;\n                    int32_t res_6973 = x_6969 + x_6971;\n                    \n                    x_6970 = res_6972;\n                    x_6971 = res_6973;\n                }\n                // write final result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_6979)[local_tid_6975] = x_6970;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_6981)[local_tid_6975] = x_6971;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // restore correct values for first block\n        {\n            if (squot32(local_tid_6975, 32) == 0) {\n                ((volatile __local\n                  int32_t *) scan_ar",
                   "r_mem_6979)[local_tid_6975] = x_6970;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_6981)[local_tid_6975] = x_6971;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // threads in bounds write partial scan result\n        {\n            if (slt32(gtid_6791, 3)) {\n                ((__global int32_t *) mem_6930)[gtid_6791] = ((__local\n                                                               int32_t *) scan_arr_mem_6979)[local_tid_6975];\n                ((__global int32_t *) mem_6933)[gtid_6791] = ((__local\n                                                               int32_t *) scan_arr_mem_6981)[local_tid_6975];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // first thread reads last element as carry-in for next iteration\n        {\n            if (local_tid_6975 == 0) {\n                x_6714 = ((__local\n                           int32_t *) scan_arr_mem_6979)[segscan_group_sizze_6787 -\n                                                         1];\n                x_6715 = ((__local\n                           int32_t *) scan_arr_mem_6981)[segscan_group_sizze_6787 -\n                                                         1];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n}\n__kernel void scan_stage1_6813(__local volatile\n                               int64_t *scan_arr_mem_7061_backing_aligned_0,\n                               __local volatile\n                               int64_t *scan_arr_mem_7063_backing_aligned_1,\n                               int32_t aoa_len_6730, __global\n                               unsigned char *mem_6936, __global\n                               unsigned char *mem_6940, __global\n                               unsigned char *mem_6943,\n                               int32_t num_threads_7046)\n{\n    const int32_t segscan_group_sizze_6808 = mainzisegscan_group_sizze_6807;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int bl",
                   "ock_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_7061_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_7061_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_7063_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_7063_backing_aligned_1;\n    int32_t global_tid_7056;\n    int32_t local_tid_7057;\n    int32_t group_sizze_7060;\n    int32_t wave_sizze_7059;\n    int32_t group_tid_7058;\n    \n    global_tid_7056 = get_global_id(0);\n    local_tid_7057 = get_local_id(0);\n    group_sizze_7060 = get_local_size(0);\n    wave_sizze_7059 = LOCKSTEP_WIDTH;\n    group_tid_7058 = get_group_id(0);\n    \n    int32_t phys_tid_6813 = global_tid_7056;\n    __local char *scan_arr_mem_7061;\n    \n    scan_arr_mem_7061 = (__local char *) scan_arr_mem_7061_backing_0;\n    \n    __local char *scan_arr_mem_7063;\n    \n    scan_arr_mem_7063 = (__local char *) scan_arr_mem_7063_backing_1;\n    \n    int32_t x_6747;\n    int32_t x_6748;\n    int32_t x_6749;\n    int32_t x_6750;\n    \n    x_6747 = 0;\n    x_6748 = 0;\n    for (int32_t j_7065 = 0; j_7065 < squot32(aoa_len_6730 + num_threads_7046 -\n                                              1, num_threads_7046); j_7065++) {\n        int32_t chunk_offset_7066 = segscan_group_sizze_6808 * j_7065 +\n                group_tid_7058 * (segscan_group_sizze_6808 *\n                                  squot32(aoa_len_6730 + num_threads_7046 - 1,\n                                          num_threads_7046));\n        int32_t flat_idx_7067 = chunk_offset_7066 + local_tid_7057;\n        int32_t gtid_6812 = flat_idx_7067;\n        \n        // threads in bounds read input; others get neutral element\n        {\n            if (slt32(gtid_6812, aoa_len_6730)) {\n                int32_t x_6756 = ((__global int32_t *) mem_6936)[gtid_6812];\n                \n                // write to-scan values to parameters\n                {\n                    x_6749 =",
                   " x_6756;\n                    x_6750 = x_6756;\n                }\n                // write mapped values results to global memory\n                { }\n            } else {\n                x_6749 = 0;\n                x_6750 = 0;\n            }\n        }\n        // combine with carry and write to local memory\n        {\n            int32_t f_6751 = x_6747 | x_6749;\n            bool cond_6752 = x_6749 == 0;\n            bool cond_6753 = !cond_6752;\n            int32_t res_6754;\n            \n            if (cond_6753) {\n                res_6754 = x_6750;\n            } else {\n                int32_t res_6755 = x_6748 + x_6750;\n                \n                res_6754 = res_6755;\n            }\n            ((__local int32_t *) scan_arr_mem_7061)[local_tid_7057] = f_6751;\n            ((__local int32_t *) scan_arr_mem_7063)[local_tid_7057] = res_6754;\n        }\n        \n        int32_t x_7047;\n        int32_t x_7048;\n        int32_t x_7049;\n        int32_t x_7050;\n        int32_t x_7068;\n        int32_t x_7069;\n        int32_t x_7070;\n        int32_t x_7071;\n        int32_t skip_threads_7077;\n        \n        if (slt32(local_tid_7057, segscan_group_sizze_6808)) {\n            x_7049 = ((volatile __local\n                       int32_t *) scan_arr_mem_7061)[local_tid_7057];\n            x_7050 = ((volatile __local\n                       int32_t *) scan_arr_mem_7063)[local_tid_7057];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_7077 = 1;\n            while (slt32(skip_threads_7077, 32)) {\n                if (sle32(skip_threads_7077, local_tid_7057 -\n                          squot32(local_tid_7057, 32) * 32) &&\n                    slt32(local_tid_7057, segscan_group_sizze_6808)) {\n                    // read operands\n                    {\n                        x_7047 = ((volatile __local\n                                   int32_t *) scan_arr_mem_7061)[local_tid_7057 -\n                                                             ",
                   "    skip_threads_7077];\n                        x_7048 = ((volatile __local\n                                   int32_t *) scan_arr_mem_7063)[local_tid_7057 -\n                                                                 skip_threads_7077];\n                    }\n                    // perform operation\n                    {\n                        int32_t f_7051 = x_7047 | x_7049;\n                        bool cond_7052 = x_7049 == 0;\n                        bool cond_7053 = !cond_7052;\n                        int32_t res_7054;\n                        \n                        if (cond_7053) {\n                            res_7054 = x_7050;\n                        } else {\n                            int32_t res_7055 = x_7048 + x_7050;\n                            \n                            res_7054 = res_7055;\n                        }\n                        x_7049 = f_7051;\n                        x_7050 = res_7054;\n                    }\n                }\n                if (sle32(wave_sizze_7059, skip_threads_7077)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_7077, local_tid_7057 -\n                          squot32(local_tid_7057, 32) * 32) &&\n                    slt32(local_tid_7057, segscan_group_sizze_6808)) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_7061)[local_tid_7057] =\n                            x_7049;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_7063)[local_tid_7057] =\n                            x_7050;\n                    }\n                }\n                if (sle32(wave_sizze_7059, skip_threads_7077)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_7077 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // last thread of block 'i' writes its result to of",
                   "fset 'i'\n        {\n            if ((local_tid_7057 - squot32(local_tid_7057, 32) * 32) == 31 &&\n                slt32(local_tid_7057, segscan_group_sizze_6808)) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_7061)[squot32(local_tid_7057, 32)] =\n                    x_7049;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_7063)[squot32(local_tid_7057, 32)] =\n                    x_7050;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n        {\n            int32_t skip_threads_7078;\n            \n            if (squot32(local_tid_7057, 32) == 0 && slt32(local_tid_7057,\n                                                          segscan_group_sizze_6808)) {\n                x_7070 = ((volatile __local\n                           int32_t *) scan_arr_mem_7061)[local_tid_7057];\n                x_7071 = ((volatile __local\n                           int32_t *) scan_arr_mem_7063)[local_tid_7057];\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_7078 = 1;\n                while (slt32(skip_threads_7078, 32)) {\n                    if (sle32(skip_threads_7078, local_tid_7057 -\n                              squot32(local_tid_7057, 32) * 32) &&\n                        (squot32(local_tid_7057, 32) == 0 &&\n                         slt32(local_tid_7057, segscan_group_sizze_6808))) {\n                        // read operands\n                        {\n                            x_7068 = ((volatile __local\n                                       int32_t *) scan_arr_mem_7061)[local_tid_7057 -\n                                                                     skip_threads_7078];\n                            x_7069 = ((volatile __local\n                                       int32_t *) scan_arr_mem_7063)[local_tid_7057 -\n                                                  ",
                   "                   skip_threads_7078];\n                        }\n                        // perform operation\n                        {\n                            int32_t f_7072 = x_7068 | x_7070;\n                            bool cond_7073 = x_7070 == 0;\n                            bool cond_7074 = !cond_7073;\n                            int32_t res_7075;\n                            \n                            if (cond_7074) {\n                                res_7075 = x_7071;\n                            } else {\n                                int32_t res_7076 = x_7069 + x_7071;\n                                \n                                res_7075 = res_7076;\n                            }\n                            x_7070 = f_7072;\n                            x_7071 = res_7075;\n                        }\n                    }\n                    if (sle32(wave_sizze_7059, skip_threads_7078)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_7078, local_tid_7057 -\n                              squot32(local_tid_7057, 32) * 32) &&\n                        (squot32(local_tid_7057, 32) == 0 &&\n                         slt32(local_tid_7057, segscan_group_sizze_6808))) {\n                        // write result\n                        {\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_7061)[local_tid_7057] =\n                                x_7070;\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_7063)[local_tid_7057] =\n                                x_7071;\n                        }\n                    }\n                    if (sle32(wave_sizze_7059, skip_threads_7078)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_7078 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for ev",
                   "ery block except the first\n        {\n            if (!(squot32(local_tid_7057, 32) == 0 || !slt32(local_tid_7057,\n                                                             segscan_group_sizze_6808))) {\n                // read operands\n                {\n                    x_7047 = ((volatile __local\n                               int32_t *) scan_arr_mem_7061)[squot32(local_tid_7057,\n                                                                     32) - 1];\n                    x_7048 = ((volatile __local\n                               int32_t *) scan_arr_mem_7063)[squot32(local_tid_7057,\n                                                                     32) - 1];\n                }\n                // perform operation\n                {\n                    int32_t f_7051 = x_7047 | x_7049;\n                    bool cond_7052 = x_7049 == 0;\n                    bool cond_7053 = !cond_7052;\n                    int32_t res_7054;\n                    \n                    if (cond_7053) {\n                        res_7054 = x_7050;\n                    } else {\n                        int32_t res_7055 = x_7048 + x_7050;\n                        \n                        res_7054 = res_7055;\n                    }\n                    x_7049 = f_7051;\n                    x_7050 = res_7054;\n                }\n                // write final result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_7061)[local_tid_7057] = x_7049;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_7063)[local_tid_7057] = x_7050;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // restore correct values for first block\n        {\n            if (squot32(local_tid_7057, 32) == 0) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_7061)[local_tid_7057] = x_7049;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_7063)[local_",
                   "tid_7057] = x_7050;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // threads in bounds write partial scan result\n        {\n            if (slt32(gtid_6812, aoa_len_6730)) {\n                ((__global int32_t *) mem_6940)[gtid_6812] = ((__local\n                                                               int32_t *) scan_arr_mem_7061)[local_tid_7057];\n                ((__global int32_t *) mem_6943)[gtid_6812] = ((__local\n                                                               int32_t *) scan_arr_mem_7063)[local_tid_7057];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // first thread reads last element as carry-in for next iteration\n        {\n            if (local_tid_7057 == 0) {\n                x_6747 = ((__local\n                           int32_t *) scan_arr_mem_7061)[segscan_group_sizze_6808 -\n                                                         1];\n                x_6748 = ((__local\n                           int32_t *) scan_arr_mem_7063)[segscan_group_sizze_6808 -\n                                                         1];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n}\n__kernel void scan_stage2_6792(__local volatile\n                               int64_t *scan_arr_mem_7011_backing_aligned_0,\n                               __local volatile\n                               int64_t *scan_arr_mem_7013_backing_aligned_1,\n                               int32_t num_groups_6789, __global\n                               unsigned char *mem_6930, __global\n                               unsigned char *mem_6933,\n                               int32_t num_threads_6967)\n{\n    const int32_t segscan_group_sizze_6787 = mainzisegscan_group_sizze_6786;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_7011_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_7011_b",
                   "acking_aligned_0;\n    __local volatile char *restrict scan_arr_mem_7013_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_7013_backing_aligned_1;\n    int32_t global_tid_7006;\n    int32_t local_tid_7007;\n    int32_t group_sizze_7010;\n    int32_t wave_sizze_7009;\n    int32_t group_tid_7008;\n    \n    global_tid_7006 = get_global_id(0);\n    local_tid_7007 = get_local_id(0);\n    group_sizze_7010 = get_local_size(0);\n    wave_sizze_7009 = LOCKSTEP_WIDTH;\n    group_tid_7008 = get_group_id(0);\n    \n    int32_t phys_tid_6792 = global_tid_7006;\n    __local char *scan_arr_mem_7011;\n    \n    scan_arr_mem_7011 = (__local char *) scan_arr_mem_7011_backing_0;\n    \n    __local char *scan_arr_mem_7013;\n    \n    scan_arr_mem_7013 = (__local char *) scan_arr_mem_7013_backing_1;\n    \n    int32_t flat_idx_7015 = (local_tid_7007 + 1) * (segscan_group_sizze_6787 *\n                                                    squot32(3 +\n                                                            num_threads_6967 -\n                                                            1,\n                                                            num_threads_6967)) -\n            1;\n    int32_t gtid_6791 = flat_idx_7015;\n    \n    // threads in bound read carries; others get neutral element\n    {\n        if (slt32(gtid_6791, 3)) {\n            ((__local int32_t *) scan_arr_mem_7011)[local_tid_7007] = ((__global\n                                                                        int32_t *) mem_6930)[gtid_6791];\n            ((__local int32_t *) scan_arr_mem_7013)[local_tid_7007] = ((__global\n                                                                        int32_t *) mem_6933)[gtid_6791];\n        } else {\n            ((__local int32_t *) scan_arr_mem_7011)[local_tid_7007] = 0;\n            ((__local int32_t *) scan_arr_mem_7013)[local_tid_7007] = 0;\n        }\n    }\n    \n    int32_t x_6994;\n    int32_t x_6995;\n    int32_t x_6996;\n    int32_t x_6997;\n    ",
                   "int32_t x_7016;\n    int32_t x_7017;\n    int32_t x_7018;\n    int32_t x_7019;\n    int32_t skip_threads_7022;\n    \n    if (slt32(local_tid_7007, num_groups_6789)) {\n        x_6996 = ((volatile __local\n                   int32_t *) scan_arr_mem_7011)[local_tid_7007];\n        x_6997 = ((volatile __local\n                   int32_t *) scan_arr_mem_7013)[local_tid_7007];\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        skip_threads_7022 = 1;\n        while (slt32(skip_threads_7022, 32)) {\n            if (sle32(skip_threads_7022, local_tid_7007 -\n                      squot32(local_tid_7007, 32) * 32) && slt32(local_tid_7007,\n                                                                 num_groups_6789)) {\n                // read operands\n                {\n                    x_6994 = ((volatile __local\n                               int32_t *) scan_arr_mem_7011)[local_tid_7007 -\n                                                             skip_threads_7022];\n                    x_6995 = ((volatile __local\n                               int32_t *) scan_arr_mem_7013)[local_tid_7007 -\n                                                             skip_threads_7022];\n                }\n                // perform operation\n                {\n                    int32_t res_6998 = x_6994 + x_6996;\n                    int32_t res_6999 = x_6995 + x_6997;\n                    \n                    x_6996 = res_6998;\n                    x_6997 = res_6999;\n                }\n            }\n            if (sle32(wave_sizze_7009, skip_threads_7022)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (sle32(skip_threads_7022, local_tid_7007 -\n                      squot32(local_tid_7007, 32) * 32) && slt32(local_tid_7007,\n                                                                 num_groups_6789)) {\n                // write result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_7011)[l",
                   "ocal_tid_7007] = x_6996;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_7013)[local_tid_7007] = x_6997;\n                }\n            }\n            if (sle32(wave_sizze_7009, skip_threads_7022)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_7022 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if ((local_tid_7007 - squot32(local_tid_7007, 32) * 32) == 31 &&\n            slt32(local_tid_7007, num_groups_6789)) {\n            ((volatile __local\n              int32_t *) scan_arr_mem_7011)[squot32(local_tid_7007, 32)] =\n                x_6996;\n            ((volatile __local\n              int32_t *) scan_arr_mem_7013)[squot32(local_tid_7007, 32)] =\n                x_6997;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n    {\n        int32_t skip_threads_7023;\n        \n        if (squot32(local_tid_7007, 32) == 0 && slt32(local_tid_7007,\n                                                      num_groups_6789)) {\n            x_7018 = ((volatile __local\n                       int32_t *) scan_arr_mem_7011)[local_tid_7007];\n            x_7019 = ((volatile __local\n                       int32_t *) scan_arr_mem_7013)[local_tid_7007];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_7023 = 1;\n            while (slt32(skip_threads_7023, 32)) {\n                if (sle32(skip_threads_7023, local_tid_7007 -\n                          squot32(local_tid_7007, 32) * 32) &&\n                    (squot32(local_tid_7007, 32) == 0 && slt32(local_tid_7007,\n                                                               num_groups_6789))) {\n                    // read operands\n                    {\n                        x_7016 = ((volatile __local\n                                   int32_t *) scan_a",
                   "rr_mem_7011)[local_tid_7007 -\n                                                                 skip_threads_7023];\n                        x_7017 = ((volatile __local\n                                   int32_t *) scan_arr_mem_7013)[local_tid_7007 -\n                                                                 skip_threads_7023];\n                    }\n                    // perform operation\n                    {\n                        int32_t res_7020 = x_7016 + x_7018;\n                        int32_t res_7021 = x_7017 + x_7019;\n                        \n                        x_7018 = res_7020;\n                        x_7019 = res_7021;\n                    }\n                }\n                if (sle32(wave_sizze_7009, skip_threads_7023)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_7023, local_tid_7007 -\n                          squot32(local_tid_7007, 32) * 32) &&\n                    (squot32(local_tid_7007, 32) == 0 && slt32(local_tid_7007,\n                                                               num_groups_6789))) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_7011)[local_tid_7007] =\n                            x_7018;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_7013)[local_tid_7007] =\n                            x_7019;\n                    }\n                }\n                if (sle32(wave_sizze_7009, skip_threads_7023)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_7023 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_7007, 32) == 0 || !slt32(local_tid_7007,\n                                                         num_groups_6789))) {\n            // read operands\n            {",
                   "\n                x_6994 = ((volatile __local\n                           int32_t *) scan_arr_mem_7011)[squot32(local_tid_7007,\n                                                                 32) - 1];\n                x_6995 = ((volatile __local\n                           int32_t *) scan_arr_mem_7013)[squot32(local_tid_7007,\n                                                                 32) - 1];\n            }\n            // perform operation\n            {\n                int32_t res_6998 = x_6994 + x_6996;\n                int32_t res_6999 = x_6995 + x_6997;\n                \n                x_6996 = res_6998;\n                x_6997 = res_6999;\n            }\n            // write final result\n            {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_7011)[local_tid_7007] = x_6996;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_7013)[local_tid_7007] = x_6997;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_7007, 32) == 0) {\n            ((volatile __local int32_t *) scan_arr_mem_7011)[local_tid_7007] =\n                x_6996;\n            ((volatile __local int32_t *) scan_arr_mem_7013)[local_tid_7007] =\n                x_6997;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // threads in bounds write scanned carries\n    {\n        if (slt32(gtid_6791, 3)) {\n            ((__global int32_t *) mem_6930)[gtid_6791] = ((__local\n                                                           int32_t *) scan_arr_mem_7011)[local_tid_7007];\n            ((__global int32_t *) mem_6933)[gtid_6791] = ((__local\n                                                           int32_t *) scan_arr_mem_7013)[local_tid_7007];\n        }\n    }\n}\n__kernel void scan_stage2_6813(__local volatile\n                               int64_t *scan_arr_mem_7102_backing_aligned_0,\n                               __local volatile\n                      ",
                   "         int64_t *scan_arr_mem_7104_backing_aligned_1,\n                               int32_t aoa_len_6730, int32_t num_groups_6810,\n                               __global unsigned char *mem_6940, __global\n                               unsigned char *mem_6943,\n                               int32_t num_threads_7046)\n{\n    const int32_t segscan_group_sizze_6808 = mainzisegscan_group_sizze_6807;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_7102_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_7102_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_7104_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_7104_backing_aligned_1;\n    int32_t global_tid_7097;\n    int32_t local_tid_7098;\n    int32_t group_sizze_7101;\n    int32_t wave_sizze_7100;\n    int32_t group_tid_7099;\n    \n    global_tid_7097 = get_global_id(0);\n    local_tid_7098 = get_local_id(0);\n    group_sizze_7101 = get_local_size(0);\n    wave_sizze_7100 = LOCKSTEP_WIDTH;\n    group_tid_7099 = get_group_id(0);\n    \n    int32_t phys_tid_6813 = global_tid_7097;\n    __local char *scan_arr_mem_7102;\n    \n    scan_arr_mem_7102 = (__local char *) scan_arr_mem_7102_backing_0;\n    \n    __local char *scan_arr_mem_7104;\n    \n    scan_arr_mem_7104 = (__local char *) scan_arr_mem_7104_backing_1;\n    \n    int32_t flat_idx_7106 = (local_tid_7098 + 1) * (segscan_group_sizze_6808 *\n                                                    squot32(aoa_len_6730 +\n                                                            num_threads_7046 -\n                                                            1,\n                                                            num_threads_7046)) -\n            1;\n    int32_t gtid_6812 = flat_idx_7106;\n    \n    // threads in bound read carries; others get neutral element\n    {\n        if ",
                   "(slt32(gtid_6812, aoa_len_6730)) {\n            ((__local int32_t *) scan_arr_mem_7102)[local_tid_7098] = ((__global\n                                                                        int32_t *) mem_6940)[gtid_6812];\n            ((__local int32_t *) scan_arr_mem_7104)[local_tid_7098] = ((__global\n                                                                        int32_t *) mem_6943)[gtid_6812];\n        } else {\n            ((__local int32_t *) scan_arr_mem_7102)[local_tid_7098] = 0;\n            ((__local int32_t *) scan_arr_mem_7104)[local_tid_7098] = 0;\n        }\n    }\n    \n    int32_t x_7079;\n    int32_t x_7080;\n    int32_t x_7081;\n    int32_t x_7082;\n    int32_t x_7107;\n    int32_t x_7108;\n    int32_t x_7109;\n    int32_t x_7110;\n    int32_t skip_threads_7116;\n    \n    if (slt32(local_tid_7098, num_groups_6810)) {\n        x_7081 = ((volatile __local\n                   int32_t *) scan_arr_mem_7102)[local_tid_7098];\n        x_7082 = ((volatile __local\n                   int32_t *) scan_arr_mem_7104)[local_tid_7098];\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        skip_threads_7116 = 1;\n        while (slt32(skip_threads_7116, 32)) {\n            if (sle32(skip_threads_7116, local_tid_7098 -\n                      squot32(local_tid_7098, 32) * 32) && slt32(local_tid_7098,\n                                                                 num_groups_6810)) {\n                // read operands\n                {\n                    x_7079 = ((volatile __local\n                               int32_t *) scan_arr_mem_7102)[local_tid_7098 -\n                                                             skip_threads_7116];\n                    x_7080 = ((volatile __local\n                               int32_t *) scan_arr_mem_7104)[local_tid_7098 -\n                                                             skip_threads_7116];\n                }\n                // perform operation\n                {\n                    int32_t f_7083 = x_7079 | x_7081;\n ",
                   "                   bool cond_7084 = x_7081 == 0;\n                    bool cond_7085 = !cond_7084;\n                    int32_t res_7086;\n                    \n                    if (cond_7085) {\n                        res_7086 = x_7082;\n                    } else {\n                        int32_t res_7087 = x_7080 + x_7082;\n                        \n                        res_7086 = res_7087;\n                    }\n                    x_7081 = f_7083;\n                    x_7082 = res_7086;\n                }\n            }\n            if (sle32(wave_sizze_7100, skip_threads_7116)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (sle32(skip_threads_7116, local_tid_7098 -\n                      squot32(local_tid_7098, 32) * 32) && slt32(local_tid_7098,\n                                                                 num_groups_6810)) {\n                // write result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_7102)[local_tid_7098] = x_7081;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_7104)[local_tid_7098] = x_7082;\n                }\n            }\n            if (sle32(wave_sizze_7100, skip_threads_7116)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_7116 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if ((local_tid_7098 - squot32(local_tid_7098, 32) * 32) == 31 &&\n            slt32(local_tid_7098, num_groups_6810)) {\n            ((volatile __local\n              int32_t *) scan_arr_mem_7102)[squot32(local_tid_7098, 32)] =\n                x_7081;\n            ((volatile __local\n              int32_t *) scan_arr_mem_7104)[squot32(local_tid_7098, 32)] =\n                x_7082;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n    {\n        ",
                   "int32_t skip_threads_7117;\n        \n        if (squot32(local_tid_7098, 32) == 0 && slt32(local_tid_7098,\n                                                      num_groups_6810)) {\n            x_7109 = ((volatile __local\n                       int32_t *) scan_arr_mem_7102)[local_tid_7098];\n            x_7110 = ((volatile __local\n                       int32_t *) scan_arr_mem_7104)[local_tid_7098];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_7117 = 1;\n            while (slt32(skip_threads_7117, 32)) {\n                if (sle32(skip_threads_7117, local_tid_7098 -\n                          squot32(local_tid_7098, 32) * 32) &&\n                    (squot32(local_tid_7098, 32) == 0 && slt32(local_tid_7098,\n                                                               num_groups_6810))) {\n                    // read operands\n                    {\n                        x_7107 = ((volatile __local\n                                   int32_t *) scan_arr_mem_7102)[local_tid_7098 -\n                                                                 skip_threads_7117];\n                        x_7108 = ((volatile __local\n                                   int32_t *) scan_arr_mem_7104)[local_tid_7098 -\n                                                                 skip_threads_7117];\n                    }\n                    // perform operation\n                    {\n                        int32_t f_7111 = x_7107 | x_7109;\n                        bool cond_7112 = x_7109 == 0;\n                        bool cond_7113 = !cond_7112;\n                        int32_t res_7114;\n                        \n                        if (cond_7113) {\n                            res_7114 = x_7110;\n                        } else {\n                            int32_t res_7115 = x_7108 + x_7110;\n                            \n                            res_7114 = res_7115;\n                        }\n                        x_7109 = f_7111;\n      ",
                   "                  x_7110 = res_7114;\n                    }\n                }\n                if (sle32(wave_sizze_7100, skip_threads_7117)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_7117, local_tid_7098 -\n                          squot32(local_tid_7098, 32) * 32) &&\n                    (squot32(local_tid_7098, 32) == 0 && slt32(local_tid_7098,\n                                                               num_groups_6810))) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_7102)[local_tid_7098] =\n                            x_7109;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_7104)[local_tid_7098] =\n                            x_7110;\n                    }\n                }\n                if (sle32(wave_sizze_7100, skip_threads_7117)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_7117 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_7098, 32) == 0 || !slt32(local_tid_7098,\n                                                         num_groups_6810))) {\n            // read operands\n            {\n                x_7079 = ((volatile __local\n                           int32_t *) scan_arr_mem_7102)[squot32(local_tid_7098,\n                                                                 32) - 1];\n                x_7080 = ((volatile __local\n                           int32_t *) scan_arr_mem_7104)[squot32(local_tid_7098,\n                                                                 32) - 1];\n            }\n            // perform operation\n            {\n                int32_t f_7083 = x_7079 | x_7081;\n                bool cond_7084 = x_7081 == 0;\n                bool cond_7085 = !cond_7084;\n          ",
                   "      int32_t res_7086;\n                \n                if (cond_7085) {\n                    res_7086 = x_7082;\n                } else {\n                    int32_t res_7087 = x_7080 + x_7082;\n                    \n                    res_7086 = res_7087;\n                }\n                x_7081 = f_7083;\n                x_7082 = res_7086;\n            }\n            // write final result\n            {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_7102)[local_tid_7098] = x_7081;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_7104)[local_tid_7098] = x_7082;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_7098, 32) == 0) {\n            ((volatile __local int32_t *) scan_arr_mem_7102)[local_tid_7098] =\n                x_7081;\n            ((volatile __local int32_t *) scan_arr_mem_7104)[local_tid_7098] =\n                x_7082;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // threads in bounds write scanned carries\n    {\n        if (slt32(gtid_6812, aoa_len_6730)) {\n            ((__global int32_t *) mem_6940)[gtid_6812] = ((__local\n                                                           int32_t *) scan_arr_mem_7102)[local_tid_7098];\n            ((__global int32_t *) mem_6943)[gtid_6812] = ((__local\n                                                           int32_t *) scan_arr_mem_7104)[local_tid_7098];\n        }\n    }\n}\n__kernel void scan_stage3_7024(__global unsigned char *mem_6930, __global\n                               unsigned char *mem_6933,\n                               int32_t num_threads_6967)\n{\n    const int32_t segscan_group_sizze_6787 = mainzisegscan_group_sizze_6786;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t scan_stage3_gtid_7024;\n    int32_t scan_stage3_ltid_7025;\n    int32_t scan_stage3_gid_7026;\n    \n    scan_stage3_gtid_7024 = ge",
                   "t_global_id(0);\n    scan_stage3_ltid_7025 = get_local_id(0);\n    scan_stage3_gid_7026 = get_group_id(0);\n    \n    int32_t phys_tid_6792 = scan_stage3_gtid_7024;\n    int32_t gtid_6791 = scan_stage3_gtid_7024;\n    int32_t orig_group_7029 = squot32(scan_stage3_gtid_7024,\n                                      segscan_group_sizze_6787 * squot32(3 +\n                                                                         num_threads_6967 -\n                                                                         1,\n                                                                         num_threads_6967));\n    int32_t carry_in_flat_idx_7030 = orig_group_7029 *\n            (segscan_group_sizze_6787 * squot32(3 + num_threads_6967 - 1,\n                                                num_threads_6967)) - 1;\n    \n    if (slt32(scan_stage3_gtid_7024, 3)) {\n        if (!(orig_group_7029 == 0 || scan_stage3_gtid_7024 ==\n              (orig_group_7029 + 1) * (segscan_group_sizze_6787 * squot32(3 +\n                                                                          num_threads_6967 -\n                                                                          1,\n                                                                          num_threads_6967)) -\n              1)) {\n            int32_t x_7000;\n            int32_t x_7001;\n            int32_t x_7002;\n            int32_t x_7003;\n            \n            x_7000 = ((__global int32_t *) mem_6930)[carry_in_flat_idx_7030];\n            x_7001 = ((__global int32_t *) mem_6933)[carry_in_flat_idx_7030];\n            x_7002 = ((__global int32_t *) mem_6930)[gtid_6791];\n            x_7003 = ((__global int32_t *) mem_6933)[gtid_6791];\n            \n            int32_t res_7004 = x_7000 + x_7002;\n            int32_t res_7005 = x_7001 + x_7003;\n            \n            x_7000 = res_7004;\n            x_7001 = res_7005;\n            ((__global int32_t *) mem_6930)[gtid_6791] = x_7000;\n            ((__global int32_t *) mem_6933)[gtid_6791] = x",
                   "_7001;\n        }\n    }\n}\n__kernel void scan_stage3_7118(int32_t aoa_len_6730, __global\n                               unsigned char *mem_6940, __global\n                               unsigned char *mem_6943,\n                               int32_t num_threads_7046)\n{\n    const int32_t segscan_group_sizze_6808 = mainzisegscan_group_sizze_6807;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t scan_stage3_gtid_7118;\n    int32_t scan_stage3_ltid_7119;\n    int32_t scan_stage3_gid_7120;\n    \n    scan_stage3_gtid_7118 = get_global_id(0);\n    scan_stage3_ltid_7119 = get_local_id(0);\n    scan_stage3_gid_7120 = get_group_id(0);\n    \n    int32_t phys_tid_6813 = scan_stage3_gtid_7118;\n    int32_t gtid_6812 = scan_stage3_gtid_7118;\n    int32_t orig_group_7123 = squot32(scan_stage3_gtid_7118,\n                                      segscan_group_sizze_6808 *\n                                      squot32(aoa_len_6730 + num_threads_7046 -\n                                              1, num_threads_7046));\n    int32_t carry_in_flat_idx_7124 = orig_group_7123 *\n            (segscan_group_sizze_6808 * squot32(aoa_len_6730 +\n                                                num_threads_7046 - 1,\n                                                num_threads_7046)) - 1;\n    \n    if (slt32(scan_stage3_gtid_7118, aoa_len_6730)) {\n        if (!(orig_group_7123 == 0 || scan_stage3_gtid_7118 ==\n              (orig_group_7123 + 1) * (segscan_group_sizze_6808 *\n                                       squot32(aoa_len_6730 + num_threads_7046 -\n                                               1, num_threads_7046)) - 1)) {\n            int32_t x_7088;\n            int32_t x_7089;\n            int32_t x_7090;\n            int32_t x_7091;\n            \n            x_7088 = ((__global int32_t *) mem_6940)[carry_in_flat_idx_7124];\n            x_7089 = ((__global int32_t *) mem_6943)[carry_in_flat_idx_7124];\n            x_7090 = ((__global int32_t *) mem_6940)[gtid",
                   "_6812];\n            x_7091 = ((__global int32_t *) mem_6943)[gtid_6812];\n            \n            int32_t f_7092 = x_7088 | x_7090;\n            bool cond_7093 = x_7090 == 0;\n            bool cond_7094 = !cond_7093;\n            int32_t res_7095;\n            \n            if (cond_7094) {\n                res_7095 = x_7091;\n            } else {\n                int32_t res_7096 = x_7089 + x_7091;\n                \n                res_7095 = res_7096;\n            }\n            x_7088 = f_7092;\n            x_7089 = res_7095;\n            ((__global int32_t *) mem_6940)[gtid_6812] = x_7088;\n            ((__global int32_t *) mem_6943)[gtid_6812] = x_7089;\n        }\n    }\n}\n__kernel void segmap_6794(int32_t aoa_len_6730, __global\n                          unsigned char *mem_6933, __global\n                          unsigned char *mem_6936)\n{\n    const int32_t segmap_group_sizze_6798 = mainzisegmap_group_sizze_6797;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_7040;\n    int32_t local_tid_7041;\n    int32_t group_sizze_7044;\n    int32_t wave_sizze_7043;\n    int32_t group_tid_7042;\n    \n    global_tid_7040 = get_global_id(0);\n    local_tid_7041 = get_local_id(0);\n    group_sizze_7044 = get_local_size(0);\n    wave_sizze_7043 = LOCKSTEP_WIDTH;\n    group_tid_7042 = get_group_id(0);\n    \n    int32_t phys_tid_6794 = global_tid_7040;\n    int32_t write_i_6793 = group_tid_7042 * segmap_group_sizze_6798 +\n            local_tid_7041;\n    \n    if (slt32(write_i_6793, 3)) {\n        int32_t x_6734 = ((__global int32_t *) mem_6933)[write_i_6793];\n        int32_t index_primexp_6906 = 1 + write_i_6793;\n        \n        if (sle32(0, x_6734) && slt32(x_6734, aoa_len_6730)) {\n            ((__global int32_t *) mem_6936)[x_6734] = index_primexp_6906;\n        }\n    }\n}\n__kernel void segmap_6855(int32_t aoa_len_6730, __global\n                          unsigned char *mem_6930, __global\n                          unsigned char *mem_6943, __g",
                   "lobal\n                          unsigned char *mem_6946)\n{\n    const int32_t segmap_group_sizze_6876 = mainzisegmap_group_sizze_6858;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_7125;\n    int32_t local_tid_7126;\n    int32_t group_sizze_7129;\n    int32_t wave_sizze_7128;\n    int32_t group_tid_7127;\n    \n    global_tid_7125 = get_global_id(0);\n    local_tid_7126 = get_local_id(0);\n    group_sizze_7129 = get_local_size(0);\n    wave_sizze_7128 = LOCKSTEP_WIDTH;\n    group_tid_7127 = get_group_id(0);\n    \n    int32_t phys_tid_6855 = global_tid_7125;\n    int32_t gtid_6854 = group_tid_7127 * segmap_group_sizze_6876 +\n            local_tid_7126;\n    \n    if (slt32(gtid_6854, aoa_len_6730)) {\n        int32_t x_6885 = ((__global int32_t *) mem_6943)[gtid_6854];\n        int32_t res_6890 = x_6885 - 1;\n        int32_t res_6891 = ((__global int32_t *) mem_6930)[res_6890];\n        \n        ((__global int32_t *) mem_6946)[gtid_6854] = res_6891;\n    }\n}\n__kernel void segmap_6893(__global unsigned char *mem_6917, __global\n                          unsigned char *mem_6920, __global\n                          unsigned char *mem_6923, __global\n                          unsigned char *mem_6946)\n{\n    const int32_t segmap_group_sizze_6897 = mainzisegmap_group_sizze_6896;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_7130;\n    int32_t local_tid_7131;\n    int32_t group_sizze_7134;\n    int32_t wave_sizze_7133;\n    int32_t group_tid_7132;\n    \n    global_tid_7130 = get_global_id(0);\n    local_tid_7131 = get_local_id(0);\n    group_sizze_7134 = get_local_size(0);\n    wave_sizze_7133 = LOCKSTEP_WIDTH;\n    group_tid_7132 = get_group_id(0);\n    \n    int32_t phys_tid_6893 = global_tid_7130;\n    int32_t write_i_6892 = group_tid_7132 * segmap_group_sizze_6897 +\n            local_tid_7131;\n    \n    if (slt32(write_i_6892, 5)) {\n        int32_t x_6780 = ((__global int32_",
                   "t *) mem_6946)[write_i_6892];\n        int32_t x_6781 = ((__global int32_t *) mem_6920)[write_i_6892];\n        int32_t write_value_6782 = ((__global\n                                     int32_t *) mem_6923)[write_i_6892];\n        int32_t res_6783 = x_6780 + x_6781;\n        \n        if (sle32(0, res_6783) && slt32(res_6783, 9)) {\n            ((__global int32_t *) mem_6917)[res_6783] = write_value_6782;\n        }\n    }\n}\n",
                   NULL};
static int32_t static_array_realtype_7139[3] = {3, 2, 4};
static int32_t static_array_realtype_7140[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
static int32_t static_array_realtype_7141[5] = {1, 2, 0, 3, 1};
static int32_t static_array_realtype_7142[5] = {10, 20, 30, 40, 50};
static int32_t static_array_realtype_7143[3] = {2, 1, 2};
struct memblock_device {
    int *references;
    cl_mem mem;
    int64_t size;
    const char *desc;
} ;
struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
static const char *size_names[] = {"main.group_size_7027",
                                   "main.group_size_7038",
                                   "main.group_size_7121",
                                   "main.segmap_group_size_6797",
                                   "main.segmap_group_size_6858",
                                   "main.segmap_group_size_6896",
                                   "main.segscan_group_size_6786",
                                   "main.segscan_group_size_6807",
                                   "main.segscan_num_groups_6788",
                                   "main.segscan_num_groups_6809"};
static const char *size_vars[] = {"mainzigroup_sizze_7027",
                                  "mainzigroup_sizze_7038",
                                  "mainzigroup_sizze_7121",
                                  "mainzisegmap_group_sizze_6797",
                                  "mainzisegmap_group_sizze_6858",
                                  "mainzisegmap_group_sizze_6896",
                                  "mainzisegscan_group_sizze_6786",
                                  "mainzisegscan_group_sizze_6807",
                                  "mainzisegscan_num_groups_6788",
                                  "mainzisegscan_num_groups_6809"};
static const char *size_classes[] = {"group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "num_groups",
                                     "num_groups"};
int futhark_get_num_sizes(void)
{
    return 10;
}
const char *futhark_get_size_name(int i)
{
    return size_names[i];
}
const char *futhark_get_size_class(int i)
{
    return size_classes[i];
}
struct sizes {
    size_t mainzigroup_sizze_7027;
    size_t mainzigroup_sizze_7038;
    size_t mainzigroup_sizze_7121;
    size_t mainzisegmap_group_sizze_6797;
    size_t mainzisegmap_group_sizze_6858;
    size_t mainzisegmap_group_sizze_6896;
    size_t mainzisegscan_group_sizze_6786;
    size_t mainzisegscan_group_sizze_6807;
    size_t mainzisegscan_num_groups_6788;
    size_t mainzisegscan_num_groups_6809;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    size_t sizes[10];
    int num_build_opts;
    const char **build_opts;
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  (struct futhark_context_config *) malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->num_build_opts = 0;
    cfg->build_opts = (const char **) malloc(sizeof(const char *));
    cfg->build_opts[0] = NULL;
    cfg->sizes[0] = 0;
    cfg->sizes[1] = 0;
    cfg->sizes[2] = 0;
    cfg->sizes[3] = 0;
    cfg->sizes[4] = 0;
    cfg->sizes[5] = 0;
    cfg->sizes[6] = 0;
    cfg->sizes[7] = 0;
    cfg->sizes[8] = 0;
    cfg->sizes[9] = 0;
    opencl_config_init(&cfg->opencl, 10, size_names, size_vars, cfg->sizes,
                       size_classes);
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg->build_opts);
    free(cfg);
}
void futhark_context_config_add_build_option(struct futhark_context_config *cfg,
                                             const char *opt)
{
    cfg->build_opts[cfg->num_build_opts] = opt;
    cfg->num_build_opts++;
    cfg->build_opts = (const char **) realloc(cfg->build_opts,
                                              (cfg->num_build_opts + 1) *
                                              sizeof(const char *));
    cfg->build_opts[cfg->num_build_opts] = NULL;
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->opencl.profiling = cfg->opencl.logging = cfg->opencl.debugging = flag;
}
void futhark_context_config_set_profiling(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->opencl.profiling = flag;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag)
{
    cfg->opencl.logging = flag;
}
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s)
{
    set_preferred_device(&cfg->opencl, s);
}
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s)
{
    set_preferred_platform(&cfg->opencl, s);
}
void futhark_context_config_select_device_interactively(struct futhark_context_config *cfg)
{
    select_device_interactively(&cfg->opencl);
}
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path)
{
    cfg->opencl.dump_program_to = path;
}
void futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                              const char *path)
{
    cfg->opencl.load_program_from = path;
}
void futhark_context_config_dump_binary_to(struct futhark_context_config *cfg,
                                           const char *path)
{
    cfg->opencl.dump_binary_to = path;
}
void futhark_context_config_load_binary_from(struct futhark_context_config *cfg,
                                             const char *path)
{
    cfg->opencl.load_binary_from = path;
}
void futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                                   int size)
{
    cfg->opencl.default_group_size = size;
    cfg->opencl.default_group_size_changed = 1;
}
void futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                                   int num)
{
    cfg->opencl.default_num_groups = num;
}
void futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_tile_size = size;
    cfg->opencl.default_tile_size_changed = 1;
}
void futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_threshold = size;
}
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value)
{
    for (int i = 0; i < 10; i++) {
        if (strcmp(size_name, size_names[i]) == 0) {
            cfg->sizes[i] = size_value;
            return 0;
        }
    }
    if (strcmp(size_name, "default_group_size") == 0) {
        cfg->opencl.default_group_size = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_num_groups") == 0) {
        cfg->opencl.default_num_groups = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_threshold") == 0) {
        cfg->opencl.default_threshold = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_tile_size") == 0) {
        cfg->opencl.default_tile_size = size_value;
        return 0;
    }
    return 1;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    int profiling;
    int profiling_paused;
    int logging;
    lock_t lock;
    char *error;
    int64_t peak_mem_usage_device;
    int64_t cur_mem_usage_device;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
    struct memblock_device static_array_6961;
    struct memblock_device static_array_6962;
    struct memblock_device static_array_6963;
    struct memblock_device static_array_6964;
    struct memblock_device static_array_6965;
    int total_runs;
    long total_runtime;
    cl_kernel replicate_7035;
    cl_kernel scan_stage1_6792;
    cl_kernel scan_stage1_6813;
    cl_kernel scan_stage2_6792;
    cl_kernel scan_stage2_6813;
    cl_kernel scan_stage3_7024;
    cl_kernel scan_stage3_7118;
    cl_kernel segmap_6794;
    cl_kernel segmap_6855;
    cl_kernel segmap_6893;
    int64_t copy_dev_to_dev_total_runtime;
    int copy_dev_to_dev_runs;
    int64_t copy_dev_to_host_total_runtime;
    int copy_dev_to_host_runs;
    int64_t copy_host_to_dev_total_runtime;
    int copy_host_to_dev_runs;
    int64_t copy_scalar_to_dev_total_runtime;
    int copy_scalar_to_dev_runs;
    int64_t copy_scalar_from_dev_total_runtime;
    int copy_scalar_from_dev_runs;
    int64_t replicate_7035_total_runtime;
    int replicate_7035_runs;
    int64_t scan_stage1_6792_total_runtime;
    int scan_stage1_6792_runs;
    int64_t scan_stage1_6813_total_runtime;
    int scan_stage1_6813_runs;
    int64_t scan_stage2_6792_total_runtime;
    int scan_stage2_6792_runs;
    int64_t scan_stage2_6813_total_runtime;
    int scan_stage2_6813_runs;
    int64_t scan_stage3_7024_total_runtime;
    int scan_stage3_7024_runs;
    int64_t scan_stage3_7118_total_runtime;
    int scan_stage3_7118_runs;
    int64_t segmap_6794_total_runtime;
    int segmap_6794_runs;
    int64_t segmap_6855_total_runtime;
    int segmap_6855_runs;
    int64_t segmap_6893_total_runtime;
    int segmap_6893_runs;
    struct opencl_context opencl;
    struct sizes sizes;
} ;
void post_opencl_setup(struct opencl_context *ctx,
                       struct opencl_device_option *option)
{
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "NVIDIA CUDA") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 32;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "AMD Accelerated Parallel Processing") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 32;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 1;
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_num_groups = 256;
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_group_size = 256;
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_tile_size = 32;
    if ((ctx->cfg.default_threshold == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_threshold = 32768;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU)
        ctx->lockstep_width = 1;
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU)
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(ctx->cfg.default_num_groups),
                        &ctx->cfg.default_num_groups, NULL);
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU)
        ctx->cfg.default_group_size = 32;
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU)
        ctx->cfg.default_tile_size = 4;
    if ((ctx->cfg.default_threshold == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU)
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(ctx->cfg.default_threshold),
                        &ctx->cfg.default_threshold, NULL);
}
static void init_context_early(struct futhark_context_config *cfg,
                               struct futhark_context *ctx)
{
    ctx->opencl.cfg = cfg->opencl;
    ctx->detail_memory = cfg->opencl.debugging;
    ctx->debugging = cfg->opencl.debugging;
    ctx->profiling = cfg->opencl.profiling;
    ctx->profiling_paused = 0;
    ctx->logging = cfg->opencl.logging;
    ctx->error = NULL;
    ctx->opencl.profiling_records_capacity = 200;
    ctx->opencl.profiling_records_used = 0;
    ctx->opencl.profiling_records =
        malloc(ctx->opencl.profiling_records_capacity *
        sizeof(struct profiling_record));
    create_lock(&ctx->lock);
    ctx->peak_mem_usage_device = 0;
    ctx->cur_mem_usage_device = 0;
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    ctx->total_runs = 0;
    ctx->total_runtime = 0;
    ctx->copy_dev_to_dev_total_runtime = 0;
    ctx->copy_dev_to_dev_runs = 0;
    ctx->copy_dev_to_host_total_runtime = 0;
    ctx->copy_dev_to_host_runs = 0;
    ctx->copy_host_to_dev_total_runtime = 0;
    ctx->copy_host_to_dev_runs = 0;
    ctx->copy_scalar_to_dev_total_runtime = 0;
    ctx->copy_scalar_to_dev_runs = 0;
    ctx->copy_scalar_from_dev_total_runtime = 0;
    ctx->copy_scalar_from_dev_runs = 0;
    ctx->replicate_7035_total_runtime = 0;
    ctx->replicate_7035_runs = 0;
    ctx->scan_stage1_6792_total_runtime = 0;
    ctx->scan_stage1_6792_runs = 0;
    ctx->scan_stage1_6813_total_runtime = 0;
    ctx->scan_stage1_6813_runs = 0;
    ctx->scan_stage2_6792_total_runtime = 0;
    ctx->scan_stage2_6792_runs = 0;
    ctx->scan_stage2_6813_total_runtime = 0;
    ctx->scan_stage2_6813_runs = 0;
    ctx->scan_stage3_7024_total_runtime = 0;
    ctx->scan_stage3_7024_runs = 0;
    ctx->scan_stage3_7118_total_runtime = 0;
    ctx->scan_stage3_7118_runs = 0;
    ctx->segmap_6794_total_runtime = 0;
    ctx->segmap_6794_runs = 0;
    ctx->segmap_6855_total_runtime = 0;
    ctx->segmap_6855_runs = 0;
    ctx->segmap_6893_total_runtime = 0;
    ctx->segmap_6893_runs = 0;
}
static int init_context_late(struct futhark_context_config *cfg,
                             struct futhark_context *ctx, cl_program prog)
{
    cl_int error;
    
    {
        ctx->replicate_7035 = clCreateKernel(prog, "replicate_7035", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "replicate_7035");
    }
    {
        ctx->scan_stage1_6792 = clCreateKernel(prog, "scan_stage1_6792",
                                               &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage1_6792");
    }
    {
        ctx->scan_stage1_6813 = clCreateKernel(prog, "scan_stage1_6813",
                                               &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage1_6813");
    }
    {
        ctx->scan_stage2_6792 = clCreateKernel(prog, "scan_stage2_6792",
                                               &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage2_6792");
    }
    {
        ctx->scan_stage2_6813 = clCreateKernel(prog, "scan_stage2_6813",
                                               &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage2_6813");
    }
    {
        ctx->scan_stage3_7024 = clCreateKernel(prog, "scan_stage3_7024",
                                               &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage3_7024");
    }
    {
        ctx->scan_stage3_7118 = clCreateKernel(prog, "scan_stage3_7118",
                                               &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage3_7118");
    }
    {
        ctx->segmap_6794 = clCreateKernel(prog, "segmap_6794", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_6794");
    }
    {
        ctx->segmap_6855 = clCreateKernel(prog, "segmap_6855", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_6855");
    }
    {
        ctx->segmap_6893 = clCreateKernel(prog, "segmap_6893", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_6893");
    }
    {
        cl_int success;
        
        ctx->static_array_6961.references = NULL;
        ctx->static_array_6961.size = 0;
        ctx->static_array_6961.mem = clCreateBuffer(ctx->opencl.ctx,
                                                    CL_MEM_READ_WRITE, (3 >
                                                                        0 ? 3 : 1) *
                                                    sizeof(int32_t), NULL,
                                                    &success);
        OPENCL_SUCCEED_OR_RETURN(success);
        if (3 > 0)
            OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                          ctx->static_array_6961.mem,
                                                          CL_TRUE, 0, 3 *
                                                          sizeof(int32_t),
                                                          static_array_realtype_7139,
                                                          0, NULL, NULL));
    }
    {
        cl_int success;
        
        ctx->static_array_6962.references = NULL;
        ctx->static_array_6962.size = 0;
        ctx->static_array_6962.mem = clCreateBuffer(ctx->opencl.ctx,
                                                    CL_MEM_READ_WRITE, (9 >
                                                                        0 ? 9 : 1) *
                                                    sizeof(int32_t), NULL,
                                                    &success);
        OPENCL_SUCCEED_OR_RETURN(success);
        if (9 > 0)
            OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                          ctx->static_array_6962.mem,
                                                          CL_TRUE, 0, 9 *
                                                          sizeof(int32_t),
                                                          static_array_realtype_7140,
                                                          0, NULL, NULL));
    }
    {
        cl_int success;
        
        ctx->static_array_6963.references = NULL;
        ctx->static_array_6963.size = 0;
        ctx->static_array_6963.mem = clCreateBuffer(ctx->opencl.ctx,
                                                    CL_MEM_READ_WRITE, (5 >
                                                                        0 ? 5 : 1) *
                                                    sizeof(int32_t), NULL,
                                                    &success);
        OPENCL_SUCCEED_OR_RETURN(success);
        if (5 > 0)
            OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                          ctx->static_array_6963.mem,
                                                          CL_TRUE, 0, 5 *
                                                          sizeof(int32_t),
                                                          static_array_realtype_7141,
                                                          0, NULL, NULL));
    }
    {
        cl_int success;
        
        ctx->static_array_6964.references = NULL;
        ctx->static_array_6964.size = 0;
        ctx->static_array_6964.mem = clCreateBuffer(ctx->opencl.ctx,
                                                    CL_MEM_READ_WRITE, (5 >
                                                                        0 ? 5 : 1) *
                                                    sizeof(int32_t), NULL,
                                                    &success);
        OPENCL_SUCCEED_OR_RETURN(success);
        if (5 > 0)
            OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                          ctx->static_array_6964.mem,
                                                          CL_TRUE, 0, 5 *
                                                          sizeof(int32_t),
                                                          static_array_realtype_7142,
                                                          0, NULL, NULL));
    }
    {
        cl_int success;
        
        ctx->static_array_6965.references = NULL;
        ctx->static_array_6965.size = 0;
        ctx->static_array_6965.mem = clCreateBuffer(ctx->opencl.ctx,
                                                    CL_MEM_READ_WRITE, (3 >
                                                                        0 ? 3 : 1) *
                                                    sizeof(int32_t), NULL,
                                                    &success);
        OPENCL_SUCCEED_OR_RETURN(success);
        if (3 > 0)
            OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                          ctx->static_array_6965.mem,
                                                          CL_TRUE, 0, 3 *
                                                          sizeof(int32_t),
                                                          static_array_realtype_7143,
                                                          0, NULL, NULL));
    }
    ctx->sizes.mainzigroup_sizze_7027 = cfg->sizes[0];
    ctx->sizes.mainzigroup_sizze_7038 = cfg->sizes[1];
    ctx->sizes.mainzigroup_sizze_7121 = cfg->sizes[2];
    ctx->sizes.mainzisegmap_group_sizze_6797 = cfg->sizes[3];
    ctx->sizes.mainzisegmap_group_sizze_6858 = cfg->sizes[4];
    ctx->sizes.mainzisegmap_group_sizze_6896 = cfg->sizes[5];
    ctx->sizes.mainzisegscan_group_sizze_6786 = cfg->sizes[6];
    ctx->sizes.mainzisegscan_group_sizze_6807 = cfg->sizes[7];
    ctx->sizes.mainzisegscan_num_groups_6788 = cfg->sizes[8];
    ctx->sizes.mainzisegscan_num_groups_6809 = cfg->sizes[9];
    return 0;
}
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    
    int required_types = 0;
    
    init_context_early(cfg, ctx);
    
    cl_program prog = setup_opencl(&ctx->opencl, opencl_program, required_types,
                                   cfg->build_opts);
    
    init_context_late(cfg, ctx, prog);
    return ctx;
}
struct futhark_context *futhark_context_new_with_command_queue(struct futhark_context_config *cfg,
                                                               cl_command_queue queue)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    
    int required_types = 0;
    
    init_context_early(cfg, ctx);
    
    cl_program prog = setup_opencl_with_command_queue(&ctx->opencl, queue,
                                                      opencl_program,
                                                      required_types,
                                                      cfg->build_opts);
    
    init_context_late(cfg, ctx, prog);
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_lock(&ctx->lock);
    opencl_tally_profiling_records(&ctx->opencl);
    free(ctx->opencl.profiling_records);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    ctx->error = OPENCL_SUCCEED_NONFATAL(clFinish(ctx->opencl.queue));
    return ctx->error != NULL;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
void futhark_context_pause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 1;
}
void futhark_context_unpause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 0;
}
int futhark_context_clear_caches(struct futhark_context *ctx)
{
    ctx->error = OPENCL_SUCCEED_NONFATAL(opencl_free_all(&ctx->opencl));
    return ctx->error != NULL;
}
cl_command_queue futhark_context_get_command_queue(struct futhark_context *ctx)
{
    return ctx->opencl.queue;
}
static int memblock_unref_device(struct futhark_context *ctx,
                                 struct memblock_device *block, const
                                 char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'device'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_device -= block->size;
            OPENCL_SUCCEED_OR_RETURN(opencl_free(&ctx->opencl, block->mem,
                                                 block->desc));
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_device);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc_device(struct futhark_context *ctx,
                                 struct memblock_device *block, int64_t size,
                                 const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "space 'device'",
              ctx->cur_mem_usage_device);
    
    int ret = memblock_unref_device(ctx, block, desc);
    
    ctx->cur_mem_usage_device += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "space 'device'",
                (long long) ctx->cur_mem_usage_device);
    if (ctx->cur_mem_usage_device > ctx->peak_mem_usage_device) {
        ctx->peak_mem_usage_device = ctx->cur_mem_usage_device;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    OPENCL_SUCCEED_OR_RETURN(opencl_alloc(&ctx->opencl, size, desc,
                                          &block->mem));
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set_device(struct futhark_context *ctx,
                               struct memblock_device *lhs,
                               struct memblock_device *rhs, const
                               char *lhs_desc)
{
    int ret = memblock_unref_device(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "default space",
              ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
void futhark_debugging_report(struct futhark_context *ctx)
{
    if (ctx->detail_memory || ctx->profiling) {
        fprintf(stderr, "Peak memory usage for space 'device': %lld bytes.\n",
                (long long) ctx->peak_mem_usage_device);
        fprintf(stderr, "Peak memory usage for default space: %lld bytes.\n",
                (long long) ctx->peak_mem_usage_default);
    }
    if (ctx->profiling) {
        OPENCL_SUCCEED_FATAL(opencl_tally_profiling_records(&ctx->opencl));
        fprintf(stderr,
                "copy_dev_to_dev      ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_dev_to_dev_runs,
                (long) ctx->copy_dev_to_dev_total_runtime /
                (ctx->copy_dev_to_dev_runs !=
                 0 ? ctx->copy_dev_to_dev_runs : 1),
                (long) ctx->copy_dev_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_dev_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_dev_to_dev_runs;
        fprintf(stderr,
                "copy_dev_to_host     ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_dev_to_host_runs,
                (long) ctx->copy_dev_to_host_total_runtime /
                (ctx->copy_dev_to_host_runs !=
                 0 ? ctx->copy_dev_to_host_runs : 1),
                (long) ctx->copy_dev_to_host_total_runtime);
        ctx->total_runtime += ctx->copy_dev_to_host_total_runtime;
        ctx->total_runs += ctx->copy_dev_to_host_runs;
        fprintf(stderr,
                "copy_host_to_dev     ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_host_to_dev_runs,
                (long) ctx->copy_host_to_dev_total_runtime /
                (ctx->copy_host_to_dev_runs !=
                 0 ? ctx->copy_host_to_dev_runs : 1),
                (long) ctx->copy_host_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_host_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_host_to_dev_runs;
        fprintf(stderr,
                "copy_scalar_to_dev   ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_scalar_to_dev_runs,
                (long) ctx->copy_scalar_to_dev_total_runtime /
                (ctx->copy_scalar_to_dev_runs !=
                 0 ? ctx->copy_scalar_to_dev_runs : 1),
                (long) ctx->copy_scalar_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_scalar_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_scalar_to_dev_runs;
        fprintf(stderr,
                "copy_scalar_from_dev ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_scalar_from_dev_runs,
                (long) ctx->copy_scalar_from_dev_total_runtime /
                (ctx->copy_scalar_from_dev_runs !=
                 0 ? ctx->copy_scalar_from_dev_runs : 1),
                (long) ctx->copy_scalar_from_dev_total_runtime);
        ctx->total_runtime += ctx->copy_scalar_from_dev_total_runtime;
        ctx->total_runs += ctx->copy_scalar_from_dev_runs;
        fprintf(stderr,
                "replicate_7035       ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->replicate_7035_runs,
                (long) ctx->replicate_7035_total_runtime /
                (ctx->replicate_7035_runs != 0 ? ctx->replicate_7035_runs : 1),
                (long) ctx->replicate_7035_total_runtime);
        ctx->total_runtime += ctx->replicate_7035_total_runtime;
        ctx->total_runs += ctx->replicate_7035_runs;
        fprintf(stderr,
                "scan_stage1_6792     ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage1_6792_runs,
                (long) ctx->scan_stage1_6792_total_runtime /
                (ctx->scan_stage1_6792_runs !=
                 0 ? ctx->scan_stage1_6792_runs : 1),
                (long) ctx->scan_stage1_6792_total_runtime);
        ctx->total_runtime += ctx->scan_stage1_6792_total_runtime;
        ctx->total_runs += ctx->scan_stage1_6792_runs;
        fprintf(stderr,
                "scan_stage1_6813     ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage1_6813_runs,
                (long) ctx->scan_stage1_6813_total_runtime /
                (ctx->scan_stage1_6813_runs !=
                 0 ? ctx->scan_stage1_6813_runs : 1),
                (long) ctx->scan_stage1_6813_total_runtime);
        ctx->total_runtime += ctx->scan_stage1_6813_total_runtime;
        ctx->total_runs += ctx->scan_stage1_6813_runs;
        fprintf(stderr,
                "scan_stage2_6792     ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage2_6792_runs,
                (long) ctx->scan_stage2_6792_total_runtime /
                (ctx->scan_stage2_6792_runs !=
                 0 ? ctx->scan_stage2_6792_runs : 1),
                (long) ctx->scan_stage2_6792_total_runtime);
        ctx->total_runtime += ctx->scan_stage2_6792_total_runtime;
        ctx->total_runs += ctx->scan_stage2_6792_runs;
        fprintf(stderr,
                "scan_stage2_6813     ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage2_6813_runs,
                (long) ctx->scan_stage2_6813_total_runtime /
                (ctx->scan_stage2_6813_runs !=
                 0 ? ctx->scan_stage2_6813_runs : 1),
                (long) ctx->scan_stage2_6813_total_runtime);
        ctx->total_runtime += ctx->scan_stage2_6813_total_runtime;
        ctx->total_runs += ctx->scan_stage2_6813_runs;
        fprintf(stderr,
                "scan_stage3_7024     ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage3_7024_runs,
                (long) ctx->scan_stage3_7024_total_runtime /
                (ctx->scan_stage3_7024_runs !=
                 0 ? ctx->scan_stage3_7024_runs : 1),
                (long) ctx->scan_stage3_7024_total_runtime);
        ctx->total_runtime += ctx->scan_stage3_7024_total_runtime;
        ctx->total_runs += ctx->scan_stage3_7024_runs;
        fprintf(stderr,
                "scan_stage3_7118     ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage3_7118_runs,
                (long) ctx->scan_stage3_7118_total_runtime /
                (ctx->scan_stage3_7118_runs !=
                 0 ? ctx->scan_stage3_7118_runs : 1),
                (long) ctx->scan_stage3_7118_total_runtime);
        ctx->total_runtime += ctx->scan_stage3_7118_total_runtime;
        ctx->total_runs += ctx->scan_stage3_7118_runs;
        fprintf(stderr,
                "segmap_6794          ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_6794_runs, (long) ctx->segmap_6794_total_runtime /
                (ctx->segmap_6794_runs != 0 ? ctx->segmap_6794_runs : 1),
                (long) ctx->segmap_6794_total_runtime);
        ctx->total_runtime += ctx->segmap_6794_total_runtime;
        ctx->total_runs += ctx->segmap_6794_runs;
        fprintf(stderr,
                "segmap_6855          ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_6855_runs, (long) ctx->segmap_6855_total_runtime /
                (ctx->segmap_6855_runs != 0 ? ctx->segmap_6855_runs : 1),
                (long) ctx->segmap_6855_total_runtime);
        ctx->total_runtime += ctx->segmap_6855_total_runtime;
        ctx->total_runs += ctx->segmap_6855_runs;
        fprintf(stderr,
                "segmap_6893          ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_6893_runs, (long) ctx->segmap_6893_total_runtime /
                (ctx->segmap_6893_runs != 0 ? ctx->segmap_6893_runs : 1),
                (long) ctx->segmap_6893_total_runtime);
        ctx->total_runtime += ctx->segmap_6893_total_runtime;
        ctx->total_runs += ctx->segmap_6893_runs;
        if (ctx->profiling)
            fprintf(stderr, "%d operations with cumulative runtime: %6ldus\n",
                    ctx->total_runs, ctx->total_runtime);
    }
}
static int futrts_main(struct futhark_context *ctx,
                       struct memblock_device *out_mem_p_7135,
                       int32_t *out_out_arrsizze_7136,
                       struct memblock_device *out_mem_p_7137,
                       int32_t *out_out_arrsizze_7138);
static int futrts__replicate_i32(struct futhark_context *ctx,
                                 struct memblock_device mem_7031,
                                 int32_t num_elems_7032, int32_t val_7033);
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((uint8_t) (uint8_t) x)
#define zext_i8_i16(x) ((uint16_t) (uint8_t) x)
#define zext_i8_i32(x) ((uint32_t) (uint8_t) x)
#define zext_i8_i64(x) ((uint64_t) (uint8_t) x)
#define zext_i16_i8(x) ((uint8_t) (uint16_t) x)
#define zext_i16_i16(x) ((uint16_t) (uint16_t) x)
#define zext_i16_i32(x) ((uint32_t) (uint16_t) x)
#define zext_i16_i64(x) ((uint64_t) (uint16_t) x)
#define zext_i32_i8(x) ((uint8_t) (uint32_t) x)
#define zext_i32_i16(x) ((uint16_t) (uint32_t) x)
#define zext_i32_i32(x) ((uint32_t) (uint32_t) x)
#define zext_i32_i64(x) ((uint64_t) (uint32_t) x)
#define zext_i64_i8(x) ((uint8_t) (uint64_t) x)
#define zext_i64_i16(x) ((uint16_t) (uint64_t) x)
#define zext_i64_i32(x) ((uint32_t) (uint64_t) x)
#define zext_i64_i64(x) ((uint64_t) (uint64_t) x)
#ifdef __OPENCL_VERSION__
int32_t futrts_popc8(int8_t x)
{
    return popcount(x);
}
int32_t futrts_popc16(int16_t x)
{
    return popcount(x);
}
int32_t futrts_popc32(int32_t x)
{
    return popcount(x);
}
int32_t futrts_popc64(int64_t x)
{
    return popcount(x);
}
#elif __CUDA_ARCH__
int32_t futrts_popc8(int8_t x)
{
    return __popc(zext_i8_i32(x));
}
int32_t futrts_popc16(int16_t x)
{
    return __popc(zext_i16_i32(x));
}
int32_t futrts_popc32(int32_t x)
{
    return __popc(x);
}
int32_t futrts_popc64(int64_t x)
{
    return __popcll(x);
}
#else
int32_t futrts_popc8(int8_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
int32_t futrts_popc16(int16_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
int32_t futrts_popc32(int32_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
int32_t futrts_popc64(int64_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
#endif
#ifdef __OPENCL_VERSION__
int32_t futrts_clzz8(int8_t x)
{
    return clz(x);
}
int32_t futrts_clzz16(int16_t x)
{
    return clz(x);
}
int32_t futrts_clzz32(int32_t x)
{
    return clz(x);
}
int32_t futrts_clzz64(int64_t x)
{
    return clz(x);
}
#elif __CUDA_ARCH__
int32_t futrts_clzz8(int8_t x)
{
    return __clz(zext_i8_i32(x)) - 24;
}
int32_t futrts_clzz16(int16_t x)
{
    return __clz(zext_i16_i32(x)) - 16;
}
int32_t futrts_clzz32(int32_t x)
{
    return __clz(x);
}
int32_t futrts_clzz64(int64_t x)
{
    return __clzll(x);
}
#else
int32_t futrts_clzz8(int8_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
int32_t futrts_clzz16(int16_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
int32_t futrts_clzz32(int32_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
int32_t futrts_clzz64(int64_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
#endif
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return fmin(x, y);
}
static inline float fmax32(float x, float y)
{
    return fmax(x, y);
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return fmin(x, y);
}
static inline double fmax64(double x, double y)
{
    return fmax(x, y);
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline char cmplt64(double x, double y)
{
    return x < y;
}
static inline char cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return x;
}
static inline float fpconv_f32_f32(float x)
{
    return x;
}
static inline double fpconv_f32_f64(float x)
{
    return x;
}
static inline float fpconv_f64_f32(double x)
{
    return x;
}
static inline double fpconv_f64_f64(double x)
{
    return x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline char futrts_isnan32(float x)
{
    return isnan(x);
}
static inline char futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
#ifdef __OPENCL_VERSION__
static inline float fmod32(float x, float y)
{
    return fmod(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline float futrts_floor32(float x)
{
    return floor(x);
}
static inline float futrts_ceil32(float x)
{
    return ceil(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return mix(v0, v1, t);
}
#else
static inline float fmod32(float x, float y)
{
    return fmodf(x, y);
}
static inline float futrts_round32(float x)
{
    return rintf(x);
}
static inline float futrts_floor32(float x)
{
    return floorf(x);
}
static inline float futrts_ceil32(float x)
{
    return ceilf(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return v0 + (v1 - v0) * t;
}
#endif
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline double futrts_ceil64(double x)
{
    return ceil(x);
}
static inline double futrts_floor64(double x)
{
    return floor(x);
}
static inline char futrts_isnan64(double x)
{
    return isnan(x);
}
static inline char futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float fmod64(float x, float y)
{
    return fmod(x, y);
}
#ifdef __OPENCL_VERSION__
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return mix(v0, v1, t);
}
#else
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return v0 + (v1 - v0) * t;
}
#endif
static int futrts_main(struct futhark_context *ctx,
                       struct memblock_device *out_mem_p_7135,
                       int32_t *out_out_arrsizze_7136,
                       struct memblock_device *out_mem_p_7137,
                       int32_t *out_out_arrsizze_7138)
{
    struct memblock_device out_mem_6957;
    
    out_mem_6957.references = NULL;
    
    int32_t out_arrsizze_6958;
    struct memblock_device out_mem_6959;
    
    out_mem_6959.references = NULL;
    
    int32_t out_arrsizze_6960;
    struct memblock_device mem_6914;
    
    mem_6914.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6914, 12, "mem_6914"))
        return 1;
    
    struct memblock_device static_array_6961 = ctx->static_array_6961;
    
    if (3 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     static_array_6961.mem,
                                                     mem_6914.mem, 0, 0, 3 *
                                                     sizeof(int32_t), 0, NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_dev_to_dev_runs,
                                                                                                     &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    
    struct memblock_device mem_6917;
    
    mem_6917.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6917, 36, "mem_6917"))
        return 1;
    
    struct memblock_device static_array_6962 = ctx->static_array_6962;
    
    if (9 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     static_array_6962.mem,
                                                     mem_6917.mem, 0, 0, 9 *
                                                     sizeof(int32_t), 0, NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_dev_to_dev_runs,
                                                                                                     &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    
    struct memblock_device mem_6920;
    
    mem_6920.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6920, 20, "mem_6920"))
        return 1;
    
    struct memblock_device static_array_6963 = ctx->static_array_6963;
    
    if (5 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     static_array_6963.mem,
                                                     mem_6920.mem, 0, 0, 5 *
                                                     sizeof(int32_t), 0, NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_dev_to_dev_runs,
                                                                                                     &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    
    struct memblock_device mem_6923;
    
    mem_6923.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6923, 20, "mem_6923"))
        return 1;
    
    struct memblock_device static_array_6964 = ctx->static_array_6964;
    
    if (5 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     static_array_6964.mem,
                                                     mem_6923.mem, 0, 0, 5 *
                                                     sizeof(int32_t), 0, NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_dev_to_dev_runs,
                                                                                                     &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    
    struct memblock_device mem_6926;
    
    mem_6926.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6926, 12, "mem_6926"))
        return 1;
    
    struct memblock_device static_array_6965 = ctx->static_array_6965;
    
    if (3 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     static_array_6965.mem,
                                                     mem_6926.mem, 0, 0, 3 *
                                                     sizeof(int32_t), 0, NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_dev_to_dev_runs,
                                                                                                     &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    
    int32_t segscan_group_sizze_6787;
    
    segscan_group_sizze_6787 = ctx->sizes.mainzisegscan_group_sizze_6786;
    
    int32_t num_groups_6789;
    int32_t max_num_groups_6966;
    
    max_num_groups_6966 = ctx->sizes.mainzisegscan_num_groups_6788;
    num_groups_6789 = sext_i64_i32(smax64(1, smin64(squot64(3 +
                                                            sext_i32_i64(segscan_group_sizze_6787) -
                                                            1,
                                                            sext_i32_i64(segscan_group_sizze_6787)),
                                                    sext_i32_i64(max_num_groups_6966))));
    
    struct memblock_device mem_6930;
    
    mem_6930.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6930, 12, "mem_6930"))
        return 1;
    
    struct memblock_device mem_6933;
    
    mem_6933.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6933, 12, "mem_6933"))
        return 1;
    
    int32_t num_threads_6967 = num_groups_6789 * segscan_group_sizze_6787;
    
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6792, 0,
                                            sizeof(int32_t) *
                                            segscan_group_sizze_6787, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6792, 1,
                                            sizeof(int32_t) *
                                            segscan_group_sizze_6787, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6792, 2,
                                            sizeof(mem_6914.mem),
                                            &mem_6914.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6792, 3,
                                            sizeof(mem_6926.mem),
                                            &mem_6926.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6792, 4,
                                            sizeof(mem_6930.mem),
                                            &mem_6930.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6792, 5,
                                            sizeof(mem_6933.mem),
                                            &mem_6933.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6792, 6,
                                            sizeof(num_threads_6967),
                                            &num_threads_6967));
    if (1 * (num_groups_6789 * segscan_group_sizze_6787) != 0) {
        const size_t global_work_sizze_7144[1] = {num_groups_6789 *
                     segscan_group_sizze_6787};
        const size_t local_work_sizze_7148[1] = {segscan_group_sizze_6787};
        int64_t time_start_7145 = 0, time_end_7146 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "scan_stage1_6792");
            fprintf(stderr, "%zu", global_work_sizze_7144[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_7148[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) (0 + sizeof(int32_t) * segscan_group_sizze_6787 +
                           sizeof(int32_t) * segscan_group_sizze_6787));
            time_start_7145 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->scan_stage1_6792,
                                                        1, NULL,
                                                        global_work_sizze_7144,
                                                        local_work_sizze_7148,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->scan_stage1_6792_runs,
                                                                                                        &ctx->scan_stage1_6792_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_7146 = get_wall_time();
            
            long time_diff_7147 = time_end_7146 - time_start_7145;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "scan_stage1_6792",
                    time_diff_7147);
        }
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegScan");
    if (ctx->debugging)
        fprintf(stderr, "%s: %llu%c", "elems_per_group",
                (long long) (segscan_group_sizze_6787 * squot32(3 +
                                                                num_threads_6967 -
                                                                1,
                                                                num_threads_6967)),
                '\n');
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6792, 0,
                                            sizeof(int32_t) * num_groups_6789,
                                            NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6792, 1,
                                            sizeof(int32_t) * num_groups_6789,
                                            NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6792, 2,
                                            sizeof(num_groups_6789),
                                            &num_groups_6789));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6792, 3,
                                            sizeof(mem_6930.mem),
                                            &mem_6930.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6792, 4,
                                            sizeof(mem_6933.mem),
                                            &mem_6933.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6792, 5,
                                            sizeof(num_threads_6967),
                                            &num_threads_6967));
    if (1 * (1 * num_groups_6789) != 0) {
        const size_t global_work_sizze_7149[1] = {1 * num_groups_6789};
        const size_t local_work_sizze_7153[1] = {num_groups_6789};
        int64_t time_start_7150 = 0, time_end_7151 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "scan_stage2_6792");
            fprintf(stderr, "%zu", global_work_sizze_7149[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_7153[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) (0 + sizeof(int32_t) * num_groups_6789 +
                           sizeof(int32_t) * num_groups_6789));
            time_start_7150 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->scan_stage2_6792,
                                                        1, NULL,
                                                        global_work_sizze_7149,
                                                        local_work_sizze_7153,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->scan_stage2_6792_runs,
                                                                                                        &ctx->scan_stage2_6792_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_7151 = get_wall_time();
            
            long time_diff_7152 = time_end_7151 - time_start_7150;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "scan_stage2_6792",
                    time_diff_7152);
        }
    }
    
    int32_t group_sizze_7027;
    
    group_sizze_7027 = ctx->sizes.mainzigroup_sizze_7027;
    
    int32_t num_groups_7028;
    
    num_groups_7028 = squot32(3 + sext_i32_i32(group_sizze_7027) - 1,
                              sext_i32_i32(group_sizze_7027));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_7024, 0,
                                            sizeof(mem_6930.mem),
                                            &mem_6930.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_7024, 1,
                                            sizeof(mem_6933.mem),
                                            &mem_6933.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_7024, 2,
                                            sizeof(num_threads_6967),
                                            &num_threads_6967));
    if (1 * (num_groups_7028 * group_sizze_7027) != 0) {
        const size_t global_work_sizze_7154[1] = {num_groups_7028 *
                     group_sizze_7027};
        const size_t local_work_sizze_7158[1] = {group_sizze_7027};
        int64_t time_start_7155 = 0, time_end_7156 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "scan_stage3_7024");
            fprintf(stderr, "%zu", global_work_sizze_7154[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_7158[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_7155 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->scan_stage3_7024,
                                                        1, NULL,
                                                        global_work_sizze_7154,
                                                        local_work_sizze_7158,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->scan_stage3_7024_runs,
                                                                                                        &ctx->scan_stage3_7024_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_7156 = get_wall_time();
            
            long time_diff_7157 = time_end_7156 - time_start_7155;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "scan_stage3_7024",
                    time_diff_7157);
        }
    }
    if (memblock_unref_device(ctx, &mem_6914, "mem_6914") != 0)
        return 1;
    
    int32_t read_res_7159;
    
    OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                 mem_6933.mem, CL_TRUE, 2 *
                                                 sizeof(int32_t),
                                                 sizeof(int32_t),
                                                 &read_res_7159, 0, NULL,
                                                 ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                 &ctx->copy_scalar_from_dev_runs,
                                                                                                 &ctx->copy_scalar_from_dev_total_runtime)));
    
    int32_t x_6729 = read_res_7159;
    int32_t aoa_len_6730 = 2 + x_6729;
    int64_t binop_x_6935 = sext_i32_i64(aoa_len_6730);
    int64_t bytes_6934 = 4 * binop_x_6935;
    struct memblock_device mem_6936;
    
    mem_6936.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6936, bytes_6934, "mem_6936"))
        return 1;
    
    int call_ret_7160 = futrts__replicate_i32(ctx, mem_6936, aoa_len_6730, 0);
    
    assert(call_ret_7160 == 0);
    
    int32_t segmap_group_sizze_6798;
    
    segmap_group_sizze_6798 = ctx->sizes.mainzisegmap_group_sizze_6797;
    
    int64_t segmap_group_sizze_6799 = sext_i32_i64(segmap_group_sizze_6798);
    int64_t y_6800 = segmap_group_sizze_6799 - 1;
    int64_t x_6801 = 3 + y_6800;
    int64_t segmap_usable_groups_64_6803 = squot64(x_6801,
                                                   segmap_group_sizze_6799);
    int32_t segmap_usable_groups_6804 =
            sext_i64_i32(segmap_usable_groups_64_6803);
    
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_6794, 0,
                                            sizeof(aoa_len_6730),
                                            &aoa_len_6730));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_6794, 1,
                                            sizeof(mem_6933.mem),
                                            &mem_6933.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_6794, 2,
                                            sizeof(mem_6936.mem),
                                            &mem_6936.mem));
    if (1 * (segmap_usable_groups_6804 * segmap_group_sizze_6798) != 0) {
        const size_t global_work_sizze_7161[1] = {segmap_usable_groups_6804 *
                     segmap_group_sizze_6798};
        const size_t local_work_sizze_7165[1] = {segmap_group_sizze_6798};
        int64_t time_start_7162 = 0, time_end_7163 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "segmap_6794");
            fprintf(stderr, "%zu", global_work_sizze_7161[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_7165[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_7162 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->segmap_6794, 1,
                                                        NULL,
                                                        global_work_sizze_7161,
                                                        local_work_sizze_7165,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->segmap_6794_runs,
                                                                                                        &ctx->segmap_6794_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_7163 = get_wall_time();
            
            long time_diff_7164 = time_end_7163 - time_start_7162;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_6794",
                    time_diff_7164);
        }
    }
    if (memblock_unref_device(ctx, &mem_6933, "mem_6933") != 0)
        return 1;
    
    int32_t segscan_group_sizze_6808;
    
    segscan_group_sizze_6808 = ctx->sizes.mainzisegscan_group_sizze_6807;
    
    int32_t num_groups_6810;
    int32_t max_num_groups_7045;
    
    max_num_groups_7045 = ctx->sizes.mainzisegscan_num_groups_6809;
    num_groups_6810 = sext_i64_i32(smax64(1, smin64(squot64(binop_x_6935 +
                                                            sext_i32_i64(segscan_group_sizze_6808) -
                                                            1,
                                                            sext_i32_i64(segscan_group_sizze_6808)),
                                                    sext_i32_i64(max_num_groups_7045))));
    
    struct memblock_device mem_6940;
    
    mem_6940.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6940, bytes_6934, "mem_6940"))
        return 1;
    
    struct memblock_device mem_6943;
    
    mem_6943.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6943, bytes_6934, "mem_6943"))
        return 1;
    
    int32_t num_threads_7046 = num_groups_6810 * segscan_group_sizze_6808;
    
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6813, 0,
                                            sizeof(int32_t) *
                                            segscan_group_sizze_6808, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6813, 1,
                                            sizeof(int32_t) *
                                            segscan_group_sizze_6808, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6813, 2,
                                            sizeof(aoa_len_6730),
                                            &aoa_len_6730));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6813, 3,
                                            sizeof(mem_6936.mem),
                                            &mem_6936.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6813, 4,
                                            sizeof(mem_6940.mem),
                                            &mem_6940.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6813, 5,
                                            sizeof(mem_6943.mem),
                                            &mem_6943.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_6813, 6,
                                            sizeof(num_threads_7046),
                                            &num_threads_7046));
    if (1 * (num_groups_6810 * segscan_group_sizze_6808) != 0) {
        const size_t global_work_sizze_7166[1] = {num_groups_6810 *
                     segscan_group_sizze_6808};
        const size_t local_work_sizze_7170[1] = {segscan_group_sizze_6808};
        int64_t time_start_7167 = 0, time_end_7168 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "scan_stage1_6813");
            fprintf(stderr, "%zu", global_work_sizze_7166[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_7170[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) (0 + sizeof(int32_t) * segscan_group_sizze_6808 +
                           sizeof(int32_t) * segscan_group_sizze_6808));
            time_start_7167 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->scan_stage1_6813,
                                                        1, NULL,
                                                        global_work_sizze_7166,
                                                        local_work_sizze_7170,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->scan_stage1_6813_runs,
                                                                                                        &ctx->scan_stage1_6813_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_7168 = get_wall_time();
            
            long time_diff_7169 = time_end_7168 - time_start_7167;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "scan_stage1_6813",
                    time_diff_7169);
        }
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegScan");
    if (ctx->debugging)
        fprintf(stderr, "%s: %llu%c", "elems_per_group",
                (long long) (segscan_group_sizze_6808 * squot32(aoa_len_6730 +
                                                                num_threads_7046 -
                                                                1,
                                                                num_threads_7046)),
                '\n');
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6813, 0,
                                            sizeof(int32_t) * num_groups_6810,
                                            NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6813, 1,
                                            sizeof(int32_t) * num_groups_6810,
                                            NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6813, 2,
                                            sizeof(aoa_len_6730),
                                            &aoa_len_6730));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6813, 3,
                                            sizeof(num_groups_6810),
                                            &num_groups_6810));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6813, 4,
                                            sizeof(mem_6940.mem),
                                            &mem_6940.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6813, 5,
                                            sizeof(mem_6943.mem),
                                            &mem_6943.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_6813, 6,
                                            sizeof(num_threads_7046),
                                            &num_threads_7046));
    if (1 * (1 * num_groups_6810) != 0) {
        const size_t global_work_sizze_7171[1] = {1 * num_groups_6810};
        const size_t local_work_sizze_7175[1] = {num_groups_6810};
        int64_t time_start_7172 = 0, time_end_7173 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "scan_stage2_6813");
            fprintf(stderr, "%zu", global_work_sizze_7171[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_7175[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) (0 + sizeof(int32_t) * num_groups_6810 +
                           sizeof(int32_t) * num_groups_6810));
            time_start_7172 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->scan_stage2_6813,
                                                        1, NULL,
                                                        global_work_sizze_7171,
                                                        local_work_sizze_7175,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->scan_stage2_6813_runs,
                                                                                                        &ctx->scan_stage2_6813_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_7173 = get_wall_time();
            
            long time_diff_7174 = time_end_7173 - time_start_7172;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "scan_stage2_6813",
                    time_diff_7174);
        }
    }
    
    int32_t group_sizze_7121;
    
    group_sizze_7121 = ctx->sizes.mainzigroup_sizze_7121;
    
    int32_t num_groups_7122;
    
    num_groups_7122 = squot32(aoa_len_6730 + sext_i32_i32(group_sizze_7121) - 1,
                              sext_i32_i32(group_sizze_7121));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_7118, 0,
                                            sizeof(aoa_len_6730),
                                            &aoa_len_6730));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_7118, 1,
                                            sizeof(mem_6940.mem),
                                            &mem_6940.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_7118, 2,
                                            sizeof(mem_6943.mem),
                                            &mem_6943.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_7118, 3,
                                            sizeof(num_threads_7046),
                                            &num_threads_7046));
    if (1 * (num_groups_7122 * group_sizze_7121) != 0) {
        const size_t global_work_sizze_7176[1] = {num_groups_7122 *
                     group_sizze_7121};
        const size_t local_work_sizze_7180[1] = {group_sizze_7121};
        int64_t time_start_7177 = 0, time_end_7178 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "scan_stage3_7118");
            fprintf(stderr, "%zu", global_work_sizze_7176[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_7180[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_7177 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->scan_stage3_7118,
                                                        1, NULL,
                                                        global_work_sizze_7176,
                                                        local_work_sizze_7180,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->scan_stage3_7118_runs,
                                                                                                        &ctx->scan_stage3_7118_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_7178 = get_wall_time();
            
            long time_diff_7179 = time_end_7178 - time_start_7177;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "scan_stage3_7118",
                    time_diff_7179);
        }
    }
    if (memblock_unref_device(ctx, &mem_6936, "mem_6936") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6940, "mem_6940") != 0)
        return 1;
    
    int32_t segmap_group_sizze_6876;
    
    segmap_group_sizze_6876 = ctx->sizes.mainzisegmap_group_sizze_6858;
    
    int64_t segmap_group_sizze_6877 = sext_i32_i64(segmap_group_sizze_6876);
    int64_t y_6878 = segmap_group_sizze_6877 - 1;
    int64_t x_6879 = y_6878 + binop_x_6935;
    int64_t segmap_usable_groups_64_6881 = squot64(x_6879,
                                                   segmap_group_sizze_6877);
    int32_t segmap_usable_groups_6882 =
            sext_i64_i32(segmap_usable_groups_64_6881);
    struct memblock_device mem_6946;
    
    mem_6946.references = NULL;
    if (memblock_alloc_device(ctx, &mem_6946, bytes_6934, "mem_6946"))
        return 1;
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_6855, 0,
                                            sizeof(aoa_len_6730),
                                            &aoa_len_6730));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_6855, 1,
                                            sizeof(mem_6930.mem),
                                            &mem_6930.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_6855, 2,
                                            sizeof(mem_6943.mem),
                                            &mem_6943.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_6855, 3,
                                            sizeof(mem_6946.mem),
                                            &mem_6946.mem));
    if (1 * (segmap_usable_groups_6882 * segmap_group_sizze_6876) != 0) {
        const size_t global_work_sizze_7181[1] = {segmap_usable_groups_6882 *
                     segmap_group_sizze_6876};
        const size_t local_work_sizze_7185[1] = {segmap_group_sizze_6876};
        int64_t time_start_7182 = 0, time_end_7183 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "segmap_6855");
            fprintf(stderr, "%zu", global_work_sizze_7181[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_7185[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_7182 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->segmap_6855, 1,
                                                        NULL,
                                                        global_work_sizze_7181,
                                                        local_work_sizze_7185,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->segmap_6855_runs,
                                                                                                        &ctx->segmap_6855_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_7183 = get_wall_time();
            
            long time_diff_7184 = time_end_7183 - time_start_7182;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_6855",
                    time_diff_7184);
        }
    }
    if (memblock_unref_device(ctx, &mem_6930, "mem_6930") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6943, "mem_6943") != 0)
        return 1;
    
    int32_t segmap_group_sizze_6897;
    
    segmap_group_sizze_6897 = ctx->sizes.mainzisegmap_group_sizze_6896;
    
    int64_t segmap_group_sizze_6898 = sext_i32_i64(segmap_group_sizze_6897);
    int64_t y_6899 = segmap_group_sizze_6898 - 1;
    int64_t x_6900 = 5 + y_6899;
    int64_t segmap_usable_groups_64_6902 = squot64(x_6900,
                                                   segmap_group_sizze_6898);
    int32_t segmap_usable_groups_6903 =
            sext_i64_i32(segmap_usable_groups_64_6902);
    
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_6893, 0,
                                            sizeof(mem_6917.mem),
                                            &mem_6917.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_6893, 1,
                                            sizeof(mem_6920.mem),
                                            &mem_6920.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_6893, 2,
                                            sizeof(mem_6923.mem),
                                            &mem_6923.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_6893, 3,
                                            sizeof(mem_6946.mem),
                                            &mem_6946.mem));
    if (1 * (segmap_usable_groups_6903 * segmap_group_sizze_6897) != 0) {
        const size_t global_work_sizze_7186[1] = {segmap_usable_groups_6903 *
                     segmap_group_sizze_6897};
        const size_t local_work_sizze_7190[1] = {segmap_group_sizze_6897};
        int64_t time_start_7187 = 0, time_end_7188 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "segmap_6893");
            fprintf(stderr, "%zu", global_work_sizze_7186[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_7190[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_7187 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->segmap_6893, 1,
                                                        NULL,
                                                        global_work_sizze_7186,
                                                        local_work_sizze_7190,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->segmap_6893_runs,
                                                                                                        &ctx->segmap_6893_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_7188 = get_wall_time();
            
            long time_diff_7189 = time_end_7188 - time_start_7187;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_6893",
                    time_diff_7189);
        }
    }
    if (memblock_unref_device(ctx, &mem_6920, "mem_6920") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6923, "mem_6923") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6946, "mem_6946") != 0)
        return 1;
    out_arrsizze_6958 = 3;
    out_arrsizze_6960 = 9;
    if (memblock_set_device(ctx, &out_mem_6957, &mem_6926, "mem_6926") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_6959, &mem_6917, "mem_6917") != 0)
        return 1;
    (*out_mem_p_7135).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_7135, &out_mem_6957,
                            "out_mem_6957") != 0)
        return 1;
    *out_out_arrsizze_7136 = out_arrsizze_6958;
    (*out_mem_p_7137).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_7137, &out_mem_6959,
                            "out_mem_6959") != 0)
        return 1;
    *out_out_arrsizze_7138 = out_arrsizze_6960;
    if (memblock_unref_device(ctx, &mem_6946, "mem_6946") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6943, "mem_6943") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6940, "mem_6940") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6936, "mem_6936") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6933, "mem_6933") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6930, "mem_6930") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6926, "mem_6926") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6923, "mem_6923") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6920, "mem_6920") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6917, "mem_6917") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_6914, "mem_6914") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_6959, "out_mem_6959") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_6957, "out_mem_6957") != 0)
        return 1;
    return 0;
}
static int futrts__replicate_i32(struct futhark_context *ctx,
                                 struct memblock_device mem_7031,
                                 int32_t num_elems_7032, int32_t val_7033)
{
    int32_t group_sizze_7038;
    
    group_sizze_7038 = ctx->sizes.mainzigroup_sizze_7038;
    
    int32_t num_groups_7039;
    
    num_groups_7039 = squot32(num_elems_7032 + sext_i32_i32(group_sizze_7038) -
                              1, sext_i32_i32(group_sizze_7038));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_7035, 0,
                                            sizeof(mem_7031.mem),
                                            &mem_7031.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_7035, 1,
                                            sizeof(num_elems_7032),
                                            &num_elems_7032));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_7035, 2,
                                            sizeof(val_7033), &val_7033));
    if (1 * (num_groups_7039 * group_sizze_7038) != 0) {
        const size_t global_work_sizze_7191[1] = {num_groups_7039 *
                     group_sizze_7038};
        const size_t local_work_sizze_7195[1] = {group_sizze_7038};
        int64_t time_start_7192 = 0, time_end_7193 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "replicate_7035");
            fprintf(stderr, "%zu", global_work_sizze_7191[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_7195[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_7192 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->replicate_7035, 1,
                                                        NULL,
                                                        global_work_sizze_7191,
                                                        local_work_sizze_7195,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->replicate_7035_runs,
                                                                                                        &ctx->replicate_7035_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_7193 = get_wall_time();
            
            long time_diff_7194 = time_end_7193 - time_start_7192;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "replicate_7035",
                    time_diff_7194);
        }
    }
    return 0;
}
struct futhark_i32_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx,
                                          int32_t *data, int64_t dim0)
{
    struct futhark_i32_1d *bad = NULL;
    struct futhark_i32_1d *arr =
                          (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(int32_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    if (dim0 * sizeof(int32_t) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      dim0 * sizeof(int32_t),
                                                      data + 0, 0, NULL,
                                                      ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                      &ctx->copy_dev_to_host_runs,
                                                                                                      &ctx->copy_dev_to_host_total_runtime)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx,
                                              cl_mem data, int offset,
                                              int64_t dim0)
{
    struct futhark_i32_1d *bad = NULL;
    struct futhark_i32_1d *arr =
                          (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(int32_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    if (dim0 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     dim0 * sizeof(int32_t), 0,
                                                     NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_dev_to_dev_runs,
                                                                                                     &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i32_1d(struct futhark_context *ctx, struct futhark_i32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data)
{
    lock_lock(&ctx->lock);
    if (arr->shape[0] * sizeof(int32_t) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem, CL_TRUE, 0,
                                                     arr->shape[0] *
                                                     sizeof(int32_t), data + 0,
                                                     0, NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_host_to_dev_runs,
                                                                                                     &ctx->copy_host_to_dev_total_runtime)));
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                 struct futhark_i32_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                              struct futhark_i32_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_i32_1d **out0,
                       struct futhark_i32_1d **out1)
{
    struct memblock_device out_mem_6957;
    
    out_mem_6957.references = NULL;
    
    int32_t out_arrsizze_6958;
    struct memblock_device out_mem_6959;
    
    out_mem_6959.references = NULL;
    
    int32_t out_arrsizze_6960;
    
    lock_lock(&ctx->lock);
    
    int ret = futrts_main(ctx, &out_mem_6957, &out_arrsizze_6958, &out_mem_6959,
                          &out_arrsizze_6960);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_6957;
        (*out0)->shape[0] = out_arrsizze_6958;
        assert((*out1 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out1)->mem = out_mem_6959;
        (*out1)->shape[0] = out_arrsizze_6960;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
